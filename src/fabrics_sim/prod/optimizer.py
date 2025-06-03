# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

""" Tools for GPU accelerated optimization.

Includes an implementation of Covariance Matrix Adaptation (Evolution Strategy) (CMA-ES). This
implementation focuses on adapting the Gaussian with fixed (but changable) step sizes and prior
distributions. It leverages the GPU through PyTorch operations.

This implementation handles both constrained and unconstrained optimization. Hard constraints are
defined by a specific "hard constraint" cost threshold provided by the objective function. If hard
constraints are violated the cost of the sample should be given cost greater than or equal to the
hard constraint threshold. Samples designate in that way of violating hard constraints are removed
by the optimizer before being processed by the weights. In that sense, it effectively implements
truncated Gaussian sampling.

Notable classes:
    AdaptiveGaussian: Represents the Gaussian distribution that's updated by the algorithm
    CmaWeightPolicy: Base class for defining weight policies transforming costs into weights and
        selecting candidates. Notable deriving classes include HarmonicWeights, LogHarmonicWeights,
        ExpWeights.
    VectorizedObjective: Base class for objective functions defining the required objective API.
    CmaOptimizer: The optimizer.

The CmaOptimizer does not own its objective or AdaptiveGaussian hypothesis distribution, but it uses
those tools to implement the optimization. It provides a step() method for stepping an
AdaptiveGaussian hypothesis, and uses that step method for a pre-defined optimize() algorithm. That
step() method is a building block of CMA-style algorithms, including step-adaptive or online
variants. Users can customize the algorithm as needed by iteratively calling step() to adapt the
hypothesis.

The unit tests demonstrate second-order convergence of this optimizer on a collection of simple
reference problems including an upside down Gaussian with highly skewed Eigenspectrum and the
Rosenbrock function.
"""

import math
from typing import Optional, Tuple
import torch
import sys


def weighted_sum(weights: torch.Tensor, tensors: torch.Tensor):
    """ Calculates the weighted sum of the provided tensors.

    The tensors can be any dimension with the first dimension being the batch dimension. The result
    is a tensor of the same dimension representing the weighted sum.

    Weights should be a one-dimensional tensor of weights. There's no normalization requirement.

    For instance, the tensors might be a batch of vectors. In that case, the tensors input would be
    N x d, where d is the dimensionality of the vectors, and the result would be a d dimensional vector. 
    If the tensors are dxd matrices then tensors should N x d x d. The result will be a dxd matrix.

    Args:
        weights:
            1 dim tensor with N elements (batch size). This provides the weights.
        tensors:
            1+n dim tensor where n >= 1. n = 1 is a batch of tensors; n = 2 is a batch of matrices.
            These are the tensors we're summing over. The first dimension is N (batch size).

    Returns:
        The weighted sum as an n-dimensional tensor. E.g. if n=1 it's a vector or n=2 it's a matrix.
    """
    repeat_indices = [1]
    repeat_indices.extend(tensors.shape[1:])
    for i in range(len(tensors[0].shape)):
        weights = weights.unsqueeze(i+1)
    weights = weights.repeat(repeat_indices)
    return torch.sum(weights * tensors, dim=0)


def batch_outer(vecs1: torch.Tensor, vecs2: torch.Tensor):
    """ Computes the batch outer product of the batch vectors in vec1 and vec2.

    Args:
        vecs1: N x d tensor of N d-dimensional vectors.
        vecs2: N x d tensor of N d-dimensional vectors.

    Returns:
        An N x d x d tensor of batch outer products of the batch vectors in vec1 and vec2.

    """
    return vecs1.unsqueeze(2) @ vecs2.unsqueeze(1)


def weighted_sum_of_outers(weights, samples):
    """ Returns the weighted sum of the outer products of the sampled vectors with themselves.

    Args:
        weights: N dim vector w of batch weights. N is the batch size.
        samples: N x d dim matrix whose rows contain the batch of samples {x_i}.

    Returns:
        Weighted sum of the batch sample outter products:
            \sum_{i=1}^N w_i x_i x_i'
    """
    return weighted_sum(weights, batch_outer(samples, samples))


class AdaptiveGaussian:
    """ Implements the distribution interface for CMA.

    Interface:
        dist.sample(num_samples)
        dist.update(weights, samples)

    """
    def __init__(self,
                 mean: torch.Tensor,
                 cov: torch.Tensor,
                 cov_prior: Optional[torch.Tensor] = None,
                 ss: float = .5,
                 reg: float = 1e-10,
                 sample_randn_once=False):
        """ Initialize this adaptive Gaussian to the specified mean and covariance.

        ss is a value between 0 and 1 defining how much of the new proposal to take. A step size of
        1 steps all the way to the new estimate; a step size of 0 takes none of the new estimate.
        Implemented as a weighted blending:

          m = ss * m_new + (1.-ss) * m
          C = ss * C_new + (1.-ss) * C

        The reg parameter additionally adds the specified amount of the cov_prior to the new covariance
        estimate. If cov_prior is None, then the identity is used. If reg is set to 0., this step will
        be skipped. This prior augmentation is also implemented as a blending:

          C = (1.-reg)* C + reg * C_prior

        (when reg != 0 and cov_prior is provided)

        Args:
            mean: the initial mean vector of the Gaussian distribution
            cov: the initial covariance matrix of the Gaussian distribution
            cov_prior: the prior covariance matrix used as described above
            ss: Step size. How far to blend toward the new Gaussian. Should be in [0, 1]
            reg: Regularizer. How strongly to pull back toward the prior. Should be non-negative.
        """
        self.mean = mean
        self.cov = cov
        self.normal01_samples = None

        self.ss = ss
        self.reg = reg

        if cov_prior is not None:
            self.cov_prior = cov_prior
        else:
            self.cov_prior = torch.eye(self.dim)

    @property
    def dim(self) -> int:
        """ The dimensionality of this multivariate Gaussian.
        """
        return len(self.mean)

    def update_step_params(self,
            ss: float,
            reg: float) -> None:
        """ Update the step parameters.

        Args:
            ss: The step size param in [0,1]
            reg: The regularizer param (non-negative)
        """
        self.ss = ss
        self.reg = reg

    def sample(self, num_samples: int, use_cached_randn_samples: bool = False) -> torch.Tensor:
        """ Sample the specified number of samples from the Gaussian represented by this class.

        Args:
            num_samples: The number of samples to sample from the current multivariate Gaussian.
            use_cached_randn_samples: If true, it will use the cached randn (normal(0,1)) samples
                from the last call if they exist as long as there are at least num_samples of them.
                If there are no cached samples it were sample new ones. If there more samples are
                requested than available it will sample more as needed. These cached samples are
                then transformed accordingly to the covariance to get samples from the Gaussian.

        Returns:
            The sampled samples as a tensor of dimensions N x d where N is the number of samples
            and d is the dimension of the Gaussian.
        """
        if use_cached_randn_samples:
            if self.normal01_samples is None:
                self.normal01_samples = torch.randn([num_samples, self.dim])
            else:
                available_samples = len(self.normal01_samples)
                if num_samples > available_samples:
                    extra_samples = torch.randn([num_samples - available_samples, self.dim])
                    self.normal01_samples = torch.cat([self.normal01_samples, extra_samples], dim=0)
        if not use_cached_randn_samples or self.normal01_samples is None:
            # Do this only once. Reduces variance between estimates.
            self.normal01_samples = torch.randn([num_samples, self.dim])

        randn_samples = self.normal01_samples[:num_samples]

        L = torch.linalg.cholesky(self.cov)
        samples = (L.repeat([num_samples, 1, 1]) @ randn_samples.unsqueeze(2)).squeeze(2)
        samples += self.mean.repeat([num_samples, 1])
        samples[0,:] = self.mean
        return samples

    def update(self, weights: torch.Tensor, samples: torch.Tensor) -> None:
        """ Update the Gaussian using the provide weights and samples.

        Equations:

            m <- ss \sum_i w_i x_i + (1-ss) m_prev
            C <- (1-reg) * ( ss \sum_i w_i (x_i - m) (x_i - m)' + (1-ss) C_prev ) + reg * C_prior
        
        Weights are expected to be pre-normalized (if desired). Normalization isn't done explicitly
        inside this method.

        Args:
            weights: The vector of weights, one for each sample. These weights should be non-negative.
            samples: A tensor containing the samples. The return value of self.sample(...) is a valid input.
        """
        new_mean = weighted_sum(weights, samples)
        shifted_samples = samples - self.mean.repeat([len(samples), 1])  # use the true mean
        new_cov = weighted_sum_of_outers(weights, shifted_samples)

        self.mean = self.ss * new_mean + (1. - self.ss) * self.mean
        self.cov = self.ss * new_cov + (1. - self.ss) * self.cov
        if self.reg > 0.:
            self.cov = (1. - self.reg) * self.cov + self.reg * self.cov_prior
        

class CmaWeightPolicy(object):
    """ Base class for CMA weight policy implementations.
    """
    def calc_weights(self, costs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Calculates the weights used by CMA from the provided vector of costs.

        If these weights need to be normalized, the derived class should implement that. The weights are
        used as returned and left unmodified.

        Args:
            costs: a vector of cost values, one for each sample.

        Returns:
            A tuple containing (weights, selected_indices). The selected_indices are a torch tensor 
            containing the indices of selected samples, and the weights are the corresponding weights
            of those selected samples in the order supplied by selected_indices.
        """
        raise NotImplementedError()


class HarmonicWeights(CmaWeightPolicy):
    """ A weight policy that selects the top half highest performing samples and chooses their weights
    proportionally to 1/(sorted_index+1).
    """
    def calc_weights(self, costs : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, sorted_indices = torch.sort(costs)
        N = len(costs)
        weights = torch.zeros(N)
        weights[sorted_indices] = torch.tensor([1./(i+1) for i in range(N)])
        median_weight = torch.median(weights)
        selected_indices = weights > median_weight
        weights = weights[selected_indices] - median_weight
        weights /= torch.sum(weights)

        return weights, selected_indices

    def __str__(self):
        return "Linear weights"


class UniformWeights(HarmonicWeights):
    """ A weight policy that selects the top half highest performing samples and assigns them
    uniform weight. This policy is more similar to the cross-entropy method.
    """
    def calc_weights(self, costs : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weights, selected_indices = super().calc_weights(costs)
        weights[:] = 1./len(weights)

        return weights, selected_indices

    def __str__(self):
        return "Uniform weights"



class LogHarmonicWeights(CmaWeightPolicy):
    """ A weight policy that selects the top half highest performing samples and chooses their weights
    proportionally to log( (.5*(N+1)) / (sorted_index+1) ).
    """
    def calc_weights(self, costs : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, sorted_indices = torch.sort(costs)
        N = len(costs)
        weights = torch.zeros(N)
        weights[sorted_indices] = torch.tensor([math.log((N+1)/(2*(i+1))) for i in range(N)])
        selected_indices = weights > 0
        weights = weights[selected_indices]
        weights /= torch.sum(weights)

        return weights, selected_indices

    def __str__(self):
        return "Log linear weights"


class ExpWeights(CmaWeightPolicy):
    """ Represents exponential weights of the form w \propto exp(-lambda cost).

    On construction, the dynamic range is supplied and lambda is chosen so that the resulting weights
    (of selected samples) will have that dynamic range (the ratio of the largest weight to the smallest
    weight).

    The weights are normalized on return.
    """
    def __init__(self, dynamic_range: float = 1000., remove_half: bool = True):
        """ Create the weights object with the supplied parameters.

        This implementation is designed to elicit competitive second-order convergence on benchmark
        problems.

        Args:
            dynamic_range: The lambda (temperature) parameter is chosen automatically so that the
                resulting weights of selected samples have this dynamic range (the ratio of the largest
                to smallest weight).
            remove_half: If True, the half of the samples with lowest cost are removed (i.e. the top
                half is selected). Otherwise, all samples are used.
        """
        self.dynamic_range = dynamic_range
        self.remove_half = remove_half

    def calc_weights(self, costs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Calculates exponential weights given the supplied costs.

        Formula:
            w = exp(-lambda costs[indices])
            w /= sum(w)  # Normalized

            indices are the selected indices. This is either all indices (if remove_half is False) or
            the indices of just the top half of the samples with lowest cost (if remove_half is True).
            The lambda parameter is selected automatically so the ratio of the max weight to the min
            weight is the provided dynamic_range.

        Args:
            costs: The costs of each sample.

        Returns:
            A tuple containing the weights and selected indices (see base class description).
        """
        _, sorted_indices = torch.sort(costs)

        selected_indices = sorted_indices
        if self.remove_half:
            selected_indices = sorted_indices[:(len(sorted_indices)//2)]

        min_cost = costs[selected_indices[0]]
        max_cost = costs[selected_indices[-1]]
        eta = math.log(self.dynamic_range) / (max_cost - min_cost)

        log_weights = -eta * costs[selected_indices]
        log_weights -= torch.max(log_weights)  # Numerically nicer
        weights = torch.exp(log_weights)

        weights /= torch.sum(weights)
        return weights, selected_indices

    def __str__(self):
        return ("Exp weights: %f," % self.dynamic_range) + ", remove half:" + str(self.remove_half)


def rms_std_devs(cov: torch.Tensor) -> float:
    """ Calculates the RMS of the standard deviations along the Eigen-directions of the provided
    covariance matrix.

    Let C \in R^{dxd} be positive semi-definite (a covariance matrix). Then

        rms(C) = sqrt(trace(C) / d) 

    Args:
        cov: Covariance matrix to analyze

    Returns:
        The RMS measure of the Covariance matrix's standard deviations using the above formula.
    """
    return torch.sqrt(torch.trace(cov)/cov.shape[0]).item()


class VectorizedObjective(object):
    """ Abstract base class for a vectorized objective function.
    """
    def eval(self, samples: torch.Tensor) -> torch.Tensor:
        """
        Args:
            samples: A (batch x dim) tensor containing the input samples to be evaluated in parallel.

        Returns:
            A tensor containing the cost evaluations of each of the provided samples.
        """
        raise NotImplementedError()

    @property
    def hard_constraint_thresh(self) -> float:
        """ Accessor for the threshold cost defining a hard constraint.

        By default, the threshold is positive infinity (no hard constraints). Objectives with strict
        hard constraints should override this method.
        """
        return float('inf')

    @property
    def dim(self) -> float:
        """ Each objective should supply the dimensionality of its inputs.

        This should be the dimensionality of a single sample.
        """
        raise NotImplementedError()


class RefiningVectorizedObjective(VectorizedObjective):
    """ A refining objective specializes a previously defined objective. All properties are
    taken from the underlying objective by default.
    """
    def __init__(self, obj: VectorizedObjective):
        self.obj = obj

    @property
    def hard_constraint_thresh(self):
        return self.obj.hard_constraint_thresh

    @property
    def dim(self):
        return self.obj.dim


class CmaOptResult:
    """ Summarizes the result of a CMA optimization.
    """
    def __init__(self, x_opt: torch.Tensor, cost_opt: float, num_cycles: int):
        """ Create the results object by collecting the args into fields of matching names.

        Args:
            x_opt: Optimal point found.
            cost_opt: Cost of x_opt.
            num_cycles: Number of cycles run by the optimizer to find x_opt.
        """
        self.x_opt = x_opt
        self.cost_opt = cost_opt
        self.num_cycles = num_cycles


class CmaOptimizer:
    """ Covariant Matrix Adaptation (Evolution Strategy) optimizer.

    Implements a robust variant of CMA optimization built around the AdaptiveGaussian. Does not
    automatically adapt the step size, but uses a simple step size implementation in the AdaptiveGaussian.
    The weight policy is specified on construction.

    This optimization class does not own the AdaptiveGaussian being updated. It just knows how to step the
    distribution given an objective and a specified number of samples. The optimize() method is an example
    of how step() can be iteratively called to optimize an objective, but step() can also be used in other
    application specific ways.
    """
    def __init__(self, weight_policy: CmaWeightPolicy):
        """ Create a CMA algorithm with the given weight policy.
        """
        self.weight_policy = weight_policy

    def step(self, objective: VectorizedObjective, distribution: AdaptiveGaussian, num_samples: int) -> None:
        """ Step the AdaptiveGaussian to model and descend the objective.

        Each step will sample the specified number of samples from the provided distribution using
        the AdaptiveGaussian's sample() method and evaluate those samples using the objective. Those
        sample evaluations are then converted to weights using the weight policy provided on
        construction, and the weights are used to calculate an updated Gaussian distribution using
        the AdaptiveGaussian's update() method.

        Args:
            objective: The objective function being optimized.
            distribution: The Gaussian hypothesis being updated.
            num_samples: The number of samples to sample from the distribution and evaluate to
                update the hypothesis.
        """
        # Samples is num_samples x num_params. A collection of samples from the normal distribution.
        samples = distribution.sample(num_samples)
        costs = objective.eval(samples)

        # Handle hard constraints
        valid_indices = costs < objective.hard_constraint_thresh
        samples = samples[valid_indices,:]
        costs = costs[valid_indices]

        weights, selected_indices = self.weight_policy.calc_weights(costs)
        if selected_indices is not None:
            samples = samples[selected_indices,:]

        distribution.update(weights, samples)

    def optimize(self,
                 objective: VectorizedObjective,
                 distribution: AdaptiveGaussian,
                 num_samples: int, 
                 num_steps: int = 100,
                 cov_norm_thresh: Optional[float] = None,
                 cov_rms_thresh: Optional[float] = None,
                 measurement_objective: Optional[VectorizedObjective] = None,
                 verbose: bool = False) -> torch.Tensor:
        """ Optimize the provided objective starting from the provided mean and covariance.

        Args:
            objective: The objective function to optimize.
            distribution: The distribution which will be adapted through iterative calls to step().
                The distribution is expected to be in an initialized state representing a prior hypothesis.
            num_samples: The number of samples to use for each step.
            num_step: The maximum number of steps to take.
            cov_norm_thresh: If this is set, the norm of the covariance is checked each cycle. If it falls
                below this threshold, the algorithm will terminate.
            cov_rms_thresh: If this is set, the RMS of the covariance is checked each cycle. If it falls
                below this threshold, the algorithm will terminate. Note that the RMS is sqrt of the
                average squared standard deviation (variance); it's in the same units as the
                samples.
            measurement_objective: Optionally, a separate measurement objective can be specified
                to be evaluated to monitor the objective progression.
`           verbose: If True, the optimizer will print progress information.
        """
        for cycle in range(num_steps):
            self.step(objective, distribution, num_samples)
            if measurement_objective is not None:
                obj_val = measurement_objective.eval(distribution.mean.unsqueeze(0)).squeeze().item()
            else:
                obj_val = objective.eval(distribution.mean.unsqueeze(0)).squeeze().item()

            if verbose:
                print("cycle %d) obj val: %f, mean:" % (cycle+1, obj_val), distribution.mean,
                        "|C|:", torch.linalg.norm(distribution.cov).item(),
                        "rms(C):", rms_std_devs(distribution.cov))
                        
            if cov_norm_thresh is not None and distribution.cov.norm() <= cov_norm_thresh:
                if verbose:
                    print("<converged on |cov| <= %f>" % cov_norm_thresh)
                break
            if cov_rms_thresh is not None and rms_std_devs(distribution.cov) < cov_rms_thresh:
                if verbose:
                    print("<converged on rms(cov) <= %f>" % cov_rms_thresh)
                break

        return CmaOptResult(distribution.mean, obj_val, cycle+1)


if __name__ == "__main__":

    import unittest
    import argparse
    import sys
    import time

    import warp as wp

    parser = argparse.ArgumentParser("fabrics")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help=("Turns on the verbose flag in any given test."),
    )
    parser.add_argument(
        "--suppress_warp",
        action="store_true",
        help=("Don't initialize warp."),
    )

    args, unrecognized_args = parser.parse_known_args()


    class CmaOptimizerStandard:
        """ This is an experimental implementation of the standard optimizer described in the tutorial.

        Tutorial reference:

            https://arxiv.org/abs/1604.00772

        Currently, the method doesn't seem to be performing as well as the above simpler and more robust
        implementation.
        """
        # TODO: Add comments
        # TODO: Unit test more thoroughly.

        def __init__(self, dim, num_samples):
            self.dim = dim
            self.num_samples = num_samples

            n = self.dim

            min_num_samples = 4 + math.floor(3. * math.log(n)) # verified
            if self.num_samples < min_num_samples:
                raise RuntimeError("Num samples %d is smaller than the min required number of samples %d" % (
                    self.num_samples, min_num_samples))

            self.mean_n01 = math.sqrt(n) * (1. - 1./(4.*n) + 1./(21*n*n)) # verified

            self.reset()

        def reset(self):
            self.m = torch.zeros(self.dim)
            self.C = torch.eye(self.dim)
            self.sigma = 1.

            self.p_s = torch.zeros(self.dim)
            self.p_c = torch.zeros(self.dim)

            self.g = 0  # Generation number


        def step(self, obj):
            #self.sigma = 1.  # TODO: this fixes the convergence issue. debug

            self.g += 1  # TODO: verify that the generation index starts at 1
            n = self.dim  # Shorthand matching the CMA pseudocode

            # Block verified
            d, V = torch.linalg.eigh(self.C)  # C is symmetric positive definite, so we can use eigh
            s = torch.sqrt(d)  # The resulting eigenvalues are real.
            L = V * s.repeat([V.shape[0], 1])

            # Block verified 
            s_inv = torch.ones(len(s)) / s
            C_inv_sqrt = V .matmul(torch.diag(s_inv)).matmul(torch.transpose(V, 0, 1))
            # I_recon = C_inv_sqrt.matmul(C_inv_sqrt).matmul(self.C) # Verified close to I

            # Block verified
            z = torch.randn([self.num_samples, self.dim])
            y = (L.repeat([self.num_samples, 1, 1]) @ z.unsqueeze(2)).squeeze(2)
            x = self.m + self.sigma * y
            #print("  sigma:", self.sigma)

            costs = obj.eval(x)

            # TODO: use the weight classes.
            if True:
                _, sorted_indices = torch.sort(costs)
                weights = torch.zeros(self.num_samples)
                weights[sorted_indices] = torch.tensor([1./(i+1) for i in range(self.num_samples)])
                median_weight = torch.median(weights)

                # Extract selected weights (positive ones) and shift them by the median weight.
                selected_indices = weights > median_weight
                weights = weights[selected_indices] - median_weight
                weights /= torch.sum(weights)  # Normalize

            if False:
                _, sorted_indices = torch.sort(costs)
                lm = len(costs)
                weights = torch.zeros(lm)
                weights[sorted_indices] = torch.tensor([math.log((lm+1)/(2*(i+1))) for i in range(lm)])
                selected_indices = weights > 0
                weights = weights[selected_indices]
                weights /= torch.sum(weights)

            # Extract only the selected samples.
            y = y[selected_indices,:]  # This is the only one of the above sample tensors used below.

            # Block verified
            c_m = 1.
            mu_eff = 1./torch.sum(weights * weights)  # verified
            c_s = (mu_eff + 2.) / (n + mu_eff + 5.)
            d_s = 1. + 2. * max(0., math.sqrt((mu_eff-1.)/(n+1))-1.) + c_s
            c_c = (4.+mu_eff/n) / (n+4.+2.*mu_eff/n)
            alpha_cov =  2.
            c_1 = alpha_cov / (math.pow(n+1.3, 2)+mu_eff)
            c_mu = min(1.-c_1, alpha_cov * (mu_eff-2.+1./mu_eff) / (math.pow(n+2,2)+alpha_cov*mu_eff/2.))

            # Block verified
            mu = len(weights)
            weighted_step = weighted_sum(weights, y)  # <y>_w in the CMA pseudocode
            self.m += c_m * self.sigma * weighted_step

            # Block verified
            self.p_s = (1.-c_s) * self.p_s + math.sqrt(c_s * (2. - c_s) * mu_eff) * C_inv_sqrt * weighted_step
            p_s_norm = torch.linalg.norm(self.p_s)  # verified
            self.sigma = self.sigma * math.exp(c_s / d_s * (p_s_norm / self.mean_n01  - 1.))

            # Block verified
            self.h_s = 0.
            if p_s_norm / math.sqrt(1.-math.pow(1.-c_s, 2.*(self.g+1))) < (1.4+2./(n+1)) * self.mean_n01:
                self.h_s = 1.
            delta_h_s = 0.
            if (1.-self.h_s) * c_c * (2.-c_c) <= 1.:
                delta_h_s = 1.

            # Block verified
            # Covariance matrix adaptation.
            self.p_c = (1. - c_c) * self.p_c + self.h_s * math.sqrt(c_c * (2. - c_c) * mu_eff) * weighted_step
            # Note in the CMA pseudocode the first term has factor c_mu \sum weights. But in our case, the weights sum to 0
            # (no negative weights).
            self.C = ((1.+c_1*delta_h_s - c_1 - c_mu) * self.C
                      + c_1 * torch.outer(self.p_c, self.p_c)
                      + c_mu * weighted_sum_of_outers(weights, y))

        def opt(self, obj, num_steps=100, verbose=False):
            """ Optimize the provided obj starting from the provided mean and covariance.
            """
            for cycle in range(num_steps):
                self.step(obj)
                obj_val = obj.eval(self.m.unsqueeze(0), add_noise=False).squeeze().item()
                if verbose:
                    print("cycle %d) obj val: %f, mean:" % (cycle, obj_val), self.m,
                            "s|C|:", self.sigma * torch.linalg.norm(self.C).item())

            return self.m


    class VerboseTest(unittest.TestCase):
        @classmethod
        def setUpClass(self):
            self.verbose = args.verbose


    # Setup torch so everything's on the GPU by default.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


    class AdaptiveGaussianTest(VerboseTest):
        def test_verify_torch_gpu_default(self):
            mean = torch.zeros(3)
            if self.verbose:
                print("mean:", mean)
                print("mean.device:", mean.device)
                print("mean.device.type:", mean.device.type)
                print("mean.device.index:", mean.device.index)
            self.assertEqual("cuda", mean.device.type)

        def test_dim(self):
            torch.manual_seed(101)
            d = 3
            mean = torch.randn(d)
            cov = .5 * torch.eye(d)
            dist = AdaptiveGaussian(mean, cov)
            if self.verbose:
                print("dist.dim:", dist.dim)
            self.assertEqual(d, dist.dim)

        def test_sample(self):
            torch.manual_seed(101)

            d = 20  # Higher dimensional to test timings
            mean = torch.randn(d)
            cov = .5 * torch.eye(d)
            dist = AdaptiveGaussian(mean, cov)
            num_samples = 1000000

            start_time = time.perf_counter()
            samples = dist.sample(num_samples=num_samples)
            elapse = time.perf_counter() - start_time
            if self.verbose:
                print("samples:\n", samples)
                print("samples.shape:", samples.shape)
                print("sample time:", 1000. * elapse, "ms")
            self.assertEqual(num_samples, samples.shape[0])

            start_time = time.perf_counter()
            mean_est = torch.sum(samples, dim=0) / num_samples
            elapse = time.perf_counter() - start_time
            avg_err = torch.linalg.norm(mean_est - mean) / d
            if self.verbose:
                print("mean:", mean)
                print("mean_est:", mean_est)
                print("avg err per dim:", avg_err)
                print("sum time:", 1000. * elapse, "ms")
            self.assertLess(avg_err, .0002)

        def test_sample_cached(self):
            torch.manual_seed(101)

            d = 2
            mean = torch.randn(d)
            cov = torch.randn(d, d)
            cov = cov.matmul(cov.T)

            distribution = AdaptiveGaussian(mean=mean, cov=cov)
            samples1 = distribution.sample(5, use_cached_randn_samples=True)
            if self.verbose:
                print("samples1:\n", samples1)

            samples2 = distribution.sample(5, use_cached_randn_samples=True)
            if self.verbose:
                print("samples2:\n", samples2)

            samples3 = distribution.sample(7, use_cached_randn_samples=True)
            if self.verbose:
                print("samples3:\n", samples3)

            samples4 = distribution.sample(3, use_cached_randn_samples=True)
            if self.verbose:
                print("samples4:\n", samples4)

            samples5 = distribution.sample(8, use_cached_randn_samples=True)
            if self.verbose:
                print("samples5:\n", samples5)

            self.assertEqual(0., (samples1 - samples2).norm())
            self.assertEqual(0., (samples1 - samples3[:5]).norm())
            self.assertEqual(0., (samples1[:3] - samples4).norm())
            self.assertEqual(0., (samples3 - samples5[:7]).norm())

        def test_weighted_sum(self):
            if self.verbose:
                print("\ntesting sum of vectors")

            weights = torch.tensor([.5, .9, .1])
            samples = torch.tensor([[.3,  .1],
                                    [-.4, .3],
                                    [.2, -.4]])

            result = weighted_sum(weights, samples)
            expected_result = torch.tensor([.5*.3 + .9*(-.4)+ .1*.2,
                                            .5*.1 + .9*.3   + .1*(-.4)])
            err = torch.linalg.norm(expected_result - result).item()
            if self.verbose:
                print("result:", result)
                print("expected_result:", expected_result)
                print("err:", err)
            self.assertAlmostEqual(0., err, delta=1e-7)

            if self.verbose:
                print("\ntesting sum of matrices")

            torch.manual_seed(101)
            matrices = torch.randn(3, 2, 2)  # three 2x2 matrices
            weights = torch.randn(3)  # Three random weights
            sum_matrix = weighted_sum(weights, matrices)

            expected_sum_matrix = torch.zeros(2,2)
            for i in range(len(matrices)):
                expected_sum_matrix += weights[i] * matrices[i]

            err = torch.linalg.norm(expected_sum_matrix - sum_matrix).item()

            if self.verbose:
                print("sum_matrix:\n", sum_matrix)
                print("expected_sum_matrix:\n", expected_sum_matrix)
                print("err:", err)
            self.assertEqual(0., err)  # Should be exactly zero because everything's floats.

        def test_batch_outer(self):
            weights = torch.tensor([.5, .9, .1])
            samples = torch.tensor([[.3,  .1],
                                    [-.4, .3],
                                    [.2, -.4]])

            outers = batch_outer(samples, samples)
            expected_outers = torch.tensor([
                [[.3*.3, .3*.1],
                 [.1*.3, .1*.1]],
                [[(-.4)*(-.4), -.4*.3],
                 [.3*(-.4), .3*.3]],
                [[.2*.2, .2*(-.4)],
                 [-.4*.2, (-.4)*(-.4)]]
            ])
            err = torch.linalg.norm(expected_outers - outers)

            if self.verbose:
                print("outers:\n", outers)
                print("expected_outers:\n", expected_outers)
                print("err:", err)
            self.assertAlmostEqual(0., err, delta=1e-7)

        def test_weighted_sum_of_outers(self):
            weights = torch.tensor([.5, .9, .1])
            weights /= torch.sum(weights)
            samples = torch.tensor([[.3,  .1],
                                    [-.4, .3],
                                    [.2, -.4]])

            cov = weighted_sum_of_outers(weights, samples)

            manual_cov = torch.zeros(2, 2)
            for i in range(len(samples)):
                w = weights[i]
                O = torch.outer(samples[i], samples[i])
                manual_cov += w * O
            err = torch.linalg.norm(manual_cov - cov).item()

            if self.verbose:
                print("cov:\n", cov)
                print("manual_cov:\n", manual_cov)
                print("err:", err)
            self.assertEqual(0., err)

        def test_update(self):
            torch.manual_seed(101)

            d = 2
            mean = torch.randn(d)
            cov = torch.randn(d, d)
            cov = cov * torch.transpose(cov, 0, 1)
            cov_prior = .5 * torch.eye(d)
            dist = AdaptiveGaussian(mean, cov, cov_prior, ss=.75, reg=.1)

            weights = torch.tensor([.5, .9, .1])
            weights /= torch.sum(weights)
            samples = torch.tensor([[.3,  .1],
                                    [-.4, .3],
                                    [.2, -.4]])

            # API calculation
            dist.update(weights, samples)

            # Manual calculation
            new_mean = weighted_sum(weights, samples)

            shifted_samples = samples - mean.repeat([len(samples), 1])
            new_cov = weighted_sum_of_outers(weights, shifted_samples)
            updated_mean = .75 * new_mean + .25 * mean
            updated_cov = .75 * new_cov + .25 * cov
            updated_cov = .9 * updated_cov + .1 * cov_prior

            # Errors
            mean_err = torch.linalg.norm(updated_mean - dist.mean).item()
            cov_err = torch.linalg.norm(updated_cov - dist.cov).item()

            if self.verbose:
                print("updated_mean:\n", updated_mean)
                print("dist.mean:\n", dist.mean)
                print("mean_err:", mean_err)
                print()
                print("updated_cov:\n", updated_cov)
                print("dist.cov:\n", dist.cov)
                print("cov_err:", cov_err)

            self.assertEqual(0., mean_err)
            self.assertEqual(0., cov_err)

        def test_cov_norm(self):
            """ Some tests of covariance matrix size measurements.

            If C is a covariance matrix, its Eigenvalues give variances (squared standard
            deviations) along the Eigenvectors. The matrix norm, then squares those variances
            before summing them and square rooting the result:

                |C| = sqrt(\sum_i lambda_i^2) = sqrt(\sum_i sigma_i^4)

            This is in units of variance. Standard deviations have units of the vectors they're
            modeling. The trace gives sum of Eigenvalues, so the expression

                sqrt(trace(C) / d) = sqrt(1/d \sum_i lambda_i) = sqrt(1/d \sum_i sigma_i^2)

            This is the RMS measure of standard deviations sigma_i
            """
            dim = 3
            use_ones = False

            torch.manual_seed(101)
            C = torch.randn(dim, dim)
            C = C.matmul(C.T)
            d, V = torch.linalg.eigh(C)

            if use_ones:
                d = torch.ones(dim)
                C = V.matmul(d.diag()).matmul(V.T)

            s = torch.sqrt(d)

            if args.verbose:
                print("C:\n", C)
                print("|C|:", C.norm().item())
                print("d:", d)
                print("|d|:", d.norm().item())
                print("proposal:", torch.sqrt(d).norm().item() / math.sqrt(dim))
                print("second_proposal:", torch.sqrt(torch.trace(C)/dim).item())
                print("direct rms(s):", torch.sqrt(torch.sum(s * s) / len(s)).item())


    class NoisyObjective(RefiningVectorizedObjective):
        """ Represents a noisy version of an objective function.
        """
        def __init__(self, obj, noise_std_dev):
            super().__init__(obj)
            self.noise_std_dev = noise_std_dev

        def eval(self, X):
            Y = self.obj.eval(X)

            noise_indices = Y < self.hard_constraint_thresh
            Y[noise_indices] += self.noise_std_dev * torch.randn_like(Y[noise_indices])
            return Y
        

    class SimpleObjective(VectorizedObjective):
        """ Implements a simple objective function for testing purposes.

        The objective is an upside-down multivariate unnormalized Gaussian function with a skewed
        Covariance and shifted so the min is at 0.
        """
        def __init__(self):
            v1 = torch.tensor([2., 1.])
            v1 /= torch.linalg.norm(v1)
            v2 = torch.tensor([-1., 2.])
            v2 /= torch.linalg.norm(v2)

            l1 = 10.
            l2 = .1

            self.C = l1 * torch.outer(v1, v1) + l2 * torch.outer(v2, v2)
            self.m = torch.tensor([2., 3.])

        @property
        def dim(self):
            return len(self.m)

        def eval_scores(self, X):
            batch_size = X.shape[0]
            C = self.C.repeat([batch_size, 1, 1])
            m = self.m.repeat([batch_size, 1])

            diffs = X - m
            scores = (.5 * diffs.unsqueeze(1) @ (C @ diffs.unsqueeze(-1))).squeeze()
            return scores

        def eval(self, X):
            scores = self.eval_scores(X)
            return 1. - torch.exp(-scores)


    class HardConstrainedObjective(RefiningVectorizedObjective):
        def __init__(self, obj: VectorizedObjective, x_constraint: float, constraint_cost: float = 1000.):
            super().__init__(obj)
            self.x_constraint = x_constraint
            self.constraint_cost = constraint_cost

        def eval(self, samples: torch.Tensor) -> torch.Tensor:
            bad_indices = samples[:,0] > self.x_constraint
            costs = self.obj.eval(samples)
            costs[bad_indices.squeeze()] = self.constraint_cost
            return costs


    class RosenbrockObjective(VectorizedObjective):
        def __init__(self, dim):
            self._dim = dim

        @property
        def dim(self):
            return self._dim

        def eval(self, X, add_noise=True):
            if X.shape[1] != self.dim:
                raise RuntimeError("X.shape[1]=%d, self.dim=%f" % (X.shape[1], self.dim))

            N = X.shape[0]
            evals = torch.zeros(N)
            for i in range(X.shape[1]-1):
                evals += 100. * torch.pow(X[:,(i+1)] - torch.pow(X[:,i],2), 2) + torch.pow(torch.ones(N) - X[:,i], 2)
            return evals


    class SimpleObjectiveTest(VerboseTest):
        def manual_eval_scores(self, obj, X):
            batch_size = X.shape[0]
            manual_scores = torch.zeros(batch_size)
            for i in range(batch_size):
                x = X[i]
                d = x - obj.m
                manual_scores[i] = .5 * torch.dot(d, torch.mv(obj.C, d))
            return manual_scores

        def test_eval_scores(self):
            torch.manual_seed(101)
            obj = SimpleObjective()
            N = 5
            X = torch.randn(N, obj.dim)

            scores = obj.eval_scores(X)
            manual_scores = self.manual_eval_scores(obj, X)
            err = torch.linalg.norm(manual_scores - scores).item()

            if self.verbose:
                print("scores:\n", scores)
                print("manual_scores:\n", manual_scores)
                print("err:", err)
            self.assertEqual(0., err)

        def test_eval(self):
            torch.manual_seed(101)
            obj = SimpleObjective()
            N = 20
            X = torch.randn(N, obj.dim) + torch.Tensor([2., 3.])

            evals = obj.eval(X)
            expected_evals = torch.Tensor([
                1.0000, 1.0000, 0.5049, 0.9910, 0.2608, 1.0000, 0.9995, 1.0000, 0.9475,
                0.1411, 0.9997, 0.8864, 1.0000, 1.0000, 0.2646, 0.9430, 0.9491, 0.0300,
                0.9881, 0.2543])
                                
            if args.verbose:
                print("evals:\n", evals)

            err = (expected_evals - evals).norm().item()
            self.assertAlmostEqual(0., err, delta=1e-3)

    class CmaOptimizerTest(VerboseTest):
        def optimize_simple_objective(self, weight_policy, noise_std_dev, num_samples, err_thresh):
            if self.verbose:
                print("---------------------------------------------------")
                print("optimizing simple objective")
                print("weight policy:", weight_policy)
                print("noise std dev:", noise_std_dev)
                print("num_samples:", num_samples)
                print("---------------------------------------------------")
            torch.manual_seed(101)

            objective = SimpleObjective()

            optimizer = CmaOptimizer(weight_policy=weight_policy)
            distribution = AdaptiveGaussian(torch.zeros(objective.dim),
                                            torch.eye(objective.dim),
                                            ss=1.,
                                            reg=1e-10)
            x = optimizer.optimize(NoisyObjective(objective, noise_std_dev),
                                   distribution,
                                   num_samples=num_samples,
                                   num_steps=20,
                                   cov_rms_thresh=1e-3,
                                   measurement_objective=objective,
                                   verbose=args.verbose).x_opt
            x_expected = torch.tensor([2., 3.])
            err = (x - x_expected).norm().item()

            if self.verbose:
                print("x:", x)
                print("x_expected:", x_expected)
                print("err:", err)
            self.assertAlmostEqual(0., err, delta=err_thresh)

        def test_optimize_simple_objective(self):
            self.optimize_simple_objective(weight_policy=HarmonicWeights(),
                                           noise_std_dev=0.,
                                           num_samples=100,
                                           err_thresh=1e-3)
            self.optimize_simple_objective(weight_policy=UniformWeights(),
                                           noise_std_dev=0.,
                                           num_samples=100,
                                           err_thresh=1e-3)
            self.optimize_simple_objective(weight_policy=LogHarmonicWeights(),
                                           noise_std_dev=0.,
                                           num_samples=100,
                                           err_thresh=1e-2)
            self.optimize_simple_objective(weight_policy=ExpWeights(dynamic_range=1000., remove_half=True),
                                           noise_std_dev=0.,
                                           num_samples=100,
                                           err_thresh=1e-2)
            self.optimize_simple_objective(weight_policy=HarmonicWeights(),
                                           noise_std_dev=.1,
                                           num_samples=10000,
                                           err_thresh=1e-2)

        def test_hard_constrained_optimization(self):
            # Result of a long opt: obj val: 0.058328, mean: tensor([1.5000, 3.9522])
            torch.manual_seed(101)

            objective = HardConstrainedObjective(SimpleObjective(), x_constraint=1.5)
            optimizer = CmaOptimizer(weight_policy=HarmonicWeights())
            distribution = AdaptiveGaussian(torch.zeros(objective.dim), torch.eye(objective.dim), ss=1., reg=1e-8)

            x = optimizer.optimize(objective,
                                   distribution,
                                   num_samples=100,
                                   num_steps=20,
                                   cov_rms_thresh=1e-3,
                                   verbose=args.verbose).x_opt
            x_expected = torch.tensor([1.5000, 3.9522])
            err = (x - x_expected).norm().item()

            if self.verbose:
                print("x:", x)
                print("x_expected:", x_expected)
                print("err:", err)
            self.assertAlmostEqual(0., err, delta=1e-2)

        def test_standard_optimize(self):
            torch.manual_seed(101)

            noise_std_dev = .0
            obj = SimpleObjective(noise_std_dev=noise_std_dev)

            print("creating optimizer")
            optimizer = CmaOptimizerStandard(dim=obj.dim, num_samples=10)
            x = optimizer.opt(obj, num_steps=100)
            print("x:", x)

        def run_rosenbrock_optimize_small(self, weight_policy, expected_err_thresh, expected_num_cycles):
            if self.verbose:
                print("---------------------------------------------------")
                print("optimizing rosenbrock small")
                print("weight policy:", weight_policy)
                print("---------------------------------------------------")
            torch.manual_seed(101)
            torch.set_printoptions(linewidth=1000)

            dim = 2
            obj = RosenbrockObjective(dim=dim)

            num_samples = 100

            optimizer = CmaOptimizer(weight_policy=weight_policy)
            distribution = AdaptiveGaussian(torch.zeros(obj.dim), torch.eye(obj.dim), ss=1., reg=1e-10)
            res = optimizer.optimize(obj,
                                     distribution,
                                     num_samples=num_samples,
                                     num_steps=30,
                                     cov_rms_thresh=1e-4,
                                     verbose=args.verbose)
            x = res.x_opt
            x_expected = torch.tensor([1., 1.])
            err = (x - x_expected).norm().item()

            if self.verbose:
                print("x:", x)
                print("x_expected:", x_expected)
                print("err:", err)
                print("num_cycles:", res.num_cycles, "expected_num_cycles:", expected_num_cycles)
            self.assertAlmostEqual(0., err, delta=expected_err_thresh)

        def test_rosenbrock_optimize_small(self):
            self.run_rosenbrock_optimize_small(weight_policy=HarmonicWeights(),
                                               expected_err_thresh=1e-3,
                                               expected_num_cycles=11)
            self.run_rosenbrock_optimize_small(weight_policy=LogHarmonicWeights(),
                                               expected_err_thresh=1e-3,
                                               expected_num_cycles=21)
            self.run_rosenbrock_optimize_small(weight_policy=ExpWeights(),
                                               expected_err_thresh=1e-3,
                                               expected_num_cycles=17)

        def test_rosenbrock_optimize_large(self):
            torch.manual_seed(101)
            torch.set_printoptions(linewidth=1000)

            dim = 10
            obj = RosenbrockObjective(dim=dim)

            num_samples = 10000 
            weight_policy = HarmonicWeights()

            optimizer = CmaOptimizer(weight_policy=weight_policy)
            distribution = AdaptiveGaussian(torch.zeros(obj.dim), torch.eye(obj.dim), ss=.5, reg=1e-10)
            res = optimizer.optimize(obj,
                                     distribution,
                                     num_samples=num_samples,
                                     num_steps=10000,
                                     cov_norm_thresh=1e-6,
                                     verbose=self.verbose)

            expected_x_opt = torch.ones(dim)
            x_opt_err = (expected_x_opt - res.x_opt).norm().item()
            if self.verbose:
                print("res.x_opt:", res.x_opt)
                print("res.num_cycles:", res.num_cycles)
                print("x_opt_err:", x_opt_err)
            self.assertEqual(res.num_cycles, 84)
            self.assertAlmostEqual(0., x_opt_err, delta=1e-4)

        def test_standard_optimize(self):
            torch.manual_seed(101)
            torch.set_printoptions(linewidth=1000)

            dim = 2
            obj = RosenbrockObjective(dim=dim)
            optimizer = CmaOptimizerStandard(dim=dim, num_samples=100)
            x = optimizer.opt(obj, num_steps=30, verbose=args.verbose)

            if args.verbose:
                print("x:", x)

    if not args.suppress_warp:
        print("Initializing warp")
        wp.init()
        print("+++ Current cache directory: ", wp.config.kernel_cache_dir)
    else:
        print("Suppressing warp initialization")

    test_args = [sys.argv[0]] + unrecognized_args
    unittest.main(argv=test_args)
