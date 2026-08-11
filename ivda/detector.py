"""
Improved V-Detector Algorithm (IVDA), per Section 3 of the paper.

Implements the classical V-detector (negative selection with variable-radius
hyperspherical detectors) plus three togglable extensions:
  - statistical (hypothesis-test) termination            (spec section 2)
  - boundary-aware acceptance criterion                  (spec section 3)
  - (multi-class wrapper lives in multiclass.py, section 4)

All ablation variants (V0-V4) are produced by toggling `use_statistical_term`
and `use_boundary_aware` on this single generator - see spec section 5.

Coverage-estimation probes: spec section 2 originally called for probes drawn
uniformly from the ambient feature bounding box. In the PCA-15 feature space
actually used for NSL-KDD/UNSW-NB15 (see the note below),
that is infeasible - a capped detector radius captures near-zero volume of a
15-D hypercube, so uniform-box probes almost never land inside any detector
and coverage estimation never converges. Probes are instead drawn from the
real non-self *training* samples (the actual other-class data manifold),
which is both tractable and more faithful to what "coverage" should mean for
intrusion detection (covering the network-traffic manifold that intrusions
actually occupy, not empty regions of the ambient normalized feature space).
Ambient-box sampling is kept as a fallback when non_self_samples is omitted
(used only by the standalone synthetic sanity check).
"""
import numpy as np
from scipy.stats import norm
from scipy.spatial import cKDTree


class VDetectorSet:
    def __init__(
        self,
        r_min=1e-6,
        use_statistical_term=True,
        use_boundary_aware=True,
        alpha=0.05,
        n_probe=1000,
        c_target=0.999,
        rho_min=0.001,
        boundary_k=10,
        max_detectors=2000,
        max_consecutive_failures=3000,
        cap_radius=False,
        cap_multiplier=5.0,
        candidate_batch=256,
        fixed_radius=None,
        rng=None,
    ):
        self.r_min = r_min
        self.use_statistical_term = use_statistical_term
        self.use_boundary_aware = use_boundary_aware
        self.alpha = alpha
        self.n_probe = n_probe
        self.c_target = c_target
        self.rho_min = rho_min
        self.boundary_k = boundary_k
        self.max_detectors = max_detectors
        self.max_consecutive_failures = max_consecutive_failures
        self.cap_radius = cap_radius
        self.cap_multiplier = cap_multiplier
        self.candidate_batch = candidate_batch
        # fixed_radius turns the generator into the classical *fixed*-radius real-valued
        # NSA that the V-detector was introduced to improve on: every detector gets the
        # same radius, and a candidate is discarded if that radius would cover a self
        # sample. Left as None, the variable-radius V-detector rule of Eq. (1) applies.
        self.fixed_radius = fixed_radius
        self.rng = rng if rng is not None else np.random.default_rng()
        self.r_cap_ = None

        self.centers_ = None  # (m, p)
        self.radii_ = None    # (m,)

    def _boundary_self_mask(self, self_samples, non_self_samples):
        """Self points among the boundary_k self points closest to any non-self point."""
        if len(non_self_samples) == 0 or self.boundary_k <= 0:
            return np.zeros(len(self_samples), dtype=bool)
        tree = cKDTree(non_self_samples)
        min_to_nonself, _ = tree.query(self_samples, k=1)
        k = min(self.boundary_k, len(self_samples))
        boundary_idx = np.argsort(min_to_nonself)[:k]
        mask = np.zeros(len(self_samples), dtype=bool)
        mask[boundary_idx] = True
        return mask

    @staticmethod
    def _match_against(points, centers, radii):
        """Boolean mask: does any detector (centers[i], radii[i]) match each point?
        Nearest-detector-center via KDTree, then exact radius check (radii vary
        per detector, so nearest-*center* is a necessary pre-filter, not
        sufficient by itself - re-verified for correctness against the min-
        radius among near neighbors is unnecessary here since we only need
        ANY match: nearest-center is *not* guaranteed to be the matching one
        in general metric terms, so query a small k and check all of them).
        """
        if len(centers) == 0:
            return np.zeros(len(points), dtype=bool)
        c_arr, r_arr = np.array(centers), np.array(radii)
        k = min(8, len(c_arr))
        tree_c = cKDTree(c_arr)
        nbr_d, nbr_i = tree_c.query(points, k=k)
        if k == 1:
            nbr_d = nbr_d[:, None]
            nbr_i = nbr_i[:, None]
        return (nbr_d < r_arr[nbr_i]).any(axis=1)

    def _draw_probes(self, feature_lo, feature_hi, non_self_samples):
        if non_self_samples is not None and len(non_self_samples) > 0:
            idx = self.rng.integers(0, len(non_self_samples), size=self.n_probe)
            return non_self_samples[idx]
        return self.rng.uniform(feature_lo, feature_hi, size=(self.n_probe, len(feature_lo)))

    def fit(self, self_samples, feature_lo, feature_hi, non_self_samples=None):
        """
        self_samples: (n, p) samples defining the self region for this class.
        feature_lo, feature_hi: (p,) bounding box for candidate sampling.
        non_self_samples: (m, p) samples from all other classes. Used for (a)
            the boundary-aware acceptance criterion's boundary-self-point
            precomputation (spec 3b), and (b) as the coverage-estimation
            probe population (see module docstring). Falls back to ambient
            uniform-box probes if omitted.
        """
        self_samples = np.asarray(self_samples, dtype=float)
        feature_lo = np.asarray(feature_lo, dtype=float)
        feature_hi = np.asarray(feature_hi, dtype=float)
        non_self_samples = np.asarray(non_self_samples, dtype=float) if non_self_samples is not None else None

        self_tree = cKDTree(self_samples)

        if self.use_boundary_aware and non_self_samples is not None and len(non_self_samples) > 0:
            boundary_mask = self._boundary_self_mask(self_samples, non_self_samples)
        else:
            boundary_mask = np.zeros(len(self_samples), dtype=bool)

        if self.cap_radius:
            # Data-driven cap on detector radius: prevents a candidate far from
            # all self samples from producing an oversized detector that bulges
            # past the true (unsampled) self boundary on its far side - a known
            # "self-intrusion" pathology of real-valued NSA when self samples are
            # sparse relative to the space. Cap = cap_multiplier * median
            # nearest-neighbor distance among self samples themselves.
            if len(self_samples) > 1:
                nn_dist, _ = self_tree.query(self_samples, k=2)  # k=1 is the point itself (dist 0)
                self.r_cap_ = self.cap_multiplier * np.median(nn_dist[:, 1])
            else:
                self.r_cap_ = np.inf
        else:
            self.r_cap_ = np.inf

        centers = []
        radii = []
        probe_points = self._draw_probes(feature_lo, feature_hi, non_self_samples)
        probe_matched = np.zeros(self.n_probe, dtype=bool)
        probe_refresh_every = max(1, self.n_probe // 4)
        last_refresh_count = 0

        consecutive_failures = 0
        z_alpha = norm.ppf(1 - self.alpha)

        while len(centers) < self.max_detectors and consecutive_failures < self.max_consecutive_failures:
            if (self.use_boundary_aware and len(centers) > 0
                    and len(centers) - last_refresh_count >= probe_refresh_every):
                probe_points = self._draw_probes(feature_lo, feature_hi, non_self_samples)
                probe_matched = self._match_against(probe_points, centers, radii)
                last_refresh_count = len(centers)

            batch_n = min(self.candidate_batch, self.max_consecutive_failures - consecutive_failures)
            batch_n = max(batch_n, 1)
            xd_batch = self.rng.uniform(feature_lo, feature_hi, size=(batch_n, len(feature_lo)))
            nn_dist_batch, nn_idx_batch = self_tree.query(xd_batch, k=1)

            for j in range(batch_n):
                xd = xd_batch[j]
                nearest_idx = nn_idx_batch[j]
                if self.fixed_radius is not None:
                    # a uniform-radius detector is only admissible where it does not reach
                    # a self sample; radius 0 below makes it fail the r_min test and be
                    # counted as a placement failure, exactly like a self-overlapping
                    # variable-radius candidate
                    rd = self.fixed_radius if nn_dist_batch[j] >= self.fixed_radius else 0.0
                else:
                    rd = min(nn_dist_batch[j], self.r_cap_)

                if rd < self.r_min:
                    consecutive_failures += 1
                else:
                    accept = True
                    if self.use_boundary_aware:
                        probe_dists = np.linalg.norm(probe_points - xd, axis=1)
                        newly_matched = (probe_dists < rd) & (~probe_matched)
                        rho = newly_matched.sum() / self.n_probe
                        cond_a = rho >= self.rho_min
                        cond_b = boundary_mask[nearest_idx] and rho > 0
                        accept = cond_a or cond_b
                        if accept:
                            probe_matched = probe_matched | (probe_dists < rd)

                    if accept:
                        centers.append(xd)
                        radii.append(rd)
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1

                if len(centers) >= self.max_detectors or consecutive_failures >= self.max_consecutive_failures:
                    break

            if self.use_statistical_term and len(centers) > 0:
                if not self.use_boundary_aware:
                    matched = self._match_against(probe_points, centers, radii)
                else:
                    matched = probe_matched
                c_hat = matched.mean()
                denom = np.sqrt(self.c_target * (1 - self.c_target) / self.n_probe)
                z = (c_hat - self.c_target) / denom if denom > 0 else np.inf
                if z >= z_alpha:
                    break

        self.centers_ = np.array(centers) if centers else np.empty((0, len(feature_lo)))
        self.radii_ = np.array(radii) if radii else np.empty((0,))
        return self

    def match_score(self, X):
        """Phi(x) = number of detectors matching x, for each row of X."""
        X = np.asarray(X, dtype=float)
        if self.n_detectors == 0:
            return np.zeros(len(X), dtype=int)
        d = np.linalg.norm(X[:, None, :] - self.centers_[None, :, :], axis=2)
        return (d < self.radii_[None, :]).sum(axis=1)

    def predict_nonself(self, X):
        return self.match_score(X) > 0

    @property
    def n_detectors(self):
        return 0 if self.centers_ is None else len(self.centers_)
