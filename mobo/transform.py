#######################################################################################
#######################################################################################
#######################################################################################
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List, Optional, Tuple, Union

import torch
from botorch.models.transforms.utils import (
    norm_to_lognorm_mean,
    norm_to_lognorm_variance,
)
from botorch.posteriors import GPyTorchPosterior, Posterior, TransformedPosterior
from botorch.utils.transforms import normalize_indices
from linear_operator.operators import CholLinearOperator, DiagLinearOperator
from torch import Tensor
from torch.nn import Module, ModuleDict

from botorch.models.transforms import *


class OutcomeTransform(Module, ABC):
    r"""
    Abstract base class for outcome transforms.

    :meta private:
    """

    @abstractmethod
    def forward(
        self, Y: Tensor, Yvar: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Transform the outcomes in a model's training targets

        Args:
            Y: A `batch_shape x n x m`-dim tensor of training targets.
            Yvar: A `batch_shape x n x m`-dim tensor of observation noises
                associated with the training targets (if applicable).

        Returns:
            A two-tuple with the transformed outcomes:

            - The transformed outcome observations.
            - The transformed observation noise (if applicable).
        """
        pass  # pragma: no cover

    def subset_output(self, idcs: List[int]) -> OutcomeTransform:
        r"""Subset the transform along the output dimension.

        This functionality is used to properly treat outcome transformations
        in the `subset_model` functionality.

        Args:
            idcs: The output indices to subset the transform to.

        Returns:
            The current outcome transform, subset to the specified output indices.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement the "
            "`subset_output` method"
        )

    def untransform(
        self, Y: Tensor, Yvar: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Un-transform previously transformed outcomes

        Args:
            Y: A `batch_shape x n x m`-dim tensor of transfomred training targets.
            Yvar: A `batch_shape x n x m`-dim tensor of transformed observation
                noises associated with the training targets (if applicable).

        Returns:
            A two-tuple with the un-transformed outcomes:

            - The un-transformed outcome observations.
            - The un-transformed observation noise (if applicable).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement the `untransform` method"
        )

    @property
    def _is_linear(self) -> bool:
        """
        True for transformations such as `Standardize`; these should be able to apply
        `untransform_posterior` to a GPyTorchPosterior and return a GPyTorchPosterior,
        because a multivariate normal distribution should remain multivariate normal
        after applying the transform.
        """
        return False

    def untransform_posterior(self, posterior: Posterior) -> Posterior:
        r"""Un-transform a posterior.

        Posteriors with `_is_linear=True` should return a `GPyTorchPosterior` when
        `posterior` is a `GPyTorchPosterior`. Posteriors with `_is_linear=False`
        likely return a `TransformedPosterior` instead.

        Args:
            posterior: A posterior in the transformed space.

        Returns:
            The un-transformed posterior.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement the "
            "`untransform_posterior` method"
        )

class Standardize_self(OutcomeTransform):
    r"""Standardize outcomes (zero mean, unit variance).

    This module is stateful: If in train mode, calling forward updates the
    module state (i.e. the mean/std normalizing constants). If in eval mode,
    calling forward simply applies the standardization using the current module
    state.
    """

    def __init__(
        self,
        m: int,
        outputs: Optional[List[int]] = None,
        batch_shape: torch.Size = torch.Size(),  # noqa: B008
        min_stdv: float = 1e-8,
    ) -> None:
        r"""Standardize outcomes (zero mean, unit variance).

        Args:
            m: The output dimension.
            outputs: Which of the outputs to standardize. If omitted, all
                outputs will be standardized.
            batch_shape: The batch_shape of the training targets.
            min_stddv: The minimum standard deviation for which to perform
                standardization (if lower, only de-mean the data).
        """
        super().__init__()
        self.register_buffer("means", None)
        self.register_buffer("stdvs", None)
        self.register_buffer("_stdvs_sq", None)
        self._outputs = normalize_indices(outputs, d=m)
        self._m = m
        self._batch_shape = batch_shape
        self._min_stdv = min_stdv

    def forward(
        self, Y: Tensor, Yvar: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Standardize outcomes.

        If the module is in train mode, this updates the module state (i.e. the
        mean/std normalizing constants). If the module is in eval mode, simply
        applies the normalization using the module state.

        Args:
            Y: A `batch_shape x n x m`-dim tensor of training targets.
            Yvar: A `batch_shape x n x m`-dim tensor of observation noises
                associated with the training targets (if applicable).

        Returns:
            A two-tuple with the transformed outcomes:

            - The transformed outcome observations.
            - The transformed observation noise (if applicable).
        """
        if self.training:
            if Y.shape[:-2] != self._batch_shape:
                raise RuntimeError(
                    f"Expected Y.shape[:-2] to be {self._batch_shape}, matching "
                    "the `batch_shape` argument to `Standardize`, but got "
                    f"Y.shape[:-2]={Y.shape[:-2]}."
                )
            if Y.size(-1) != self._m:
                raise RuntimeError(
                    f"Wrong output dimension. Y.size(-1) is {Y.size(-1)}; expected "
                    f"{self._m}."
                )
            stdvs = Y.std(dim=-2, keepdim=True)
            stdvs = stdvs.where(stdvs >= self._min_stdv, torch.full_like(stdvs, 1.0))
            means = Y.mean(dim=-2, keepdim=True)
            if self._outputs is not None:
                unused = [i for i in range(self._m) if i not in self._outputs]
                means[..., unused] = 0.0
                stdvs[..., unused] = 1.0
            self.means = means
            self.stdvs = stdvs
            self._stdvs_sq = stdvs.pow(2)

        Y_tf = (Y - self.means) / self.stdvs
        # Yvar_tf = Yvar / self._stdvs_sq if Yvar is not None else None
        Yvar_tf = Yvar if Yvar is not None else None # for HeteroskedasticSingleTaskGP
        return Y_tf, Yvar_tf

    def subset_output(self, idcs: List[int]) -> OutcomeTransform:
        r"""Subset the transform along the output dimension.

        Args:
            idcs: The output indices to subset the transform to.

        Returns:
            The current outcome transform, subset to the specified output indices.
        """
        new_m = len(idcs)
        if new_m > self._m:
            raise RuntimeError(
                "Trying to subset a transform have more outputs than "
                " the original transform."
            )
        nlzd_idcs = normalize_indices(idcs, d=self._m)
        new_outputs = None
        if self._outputs is not None:
            new_outputs = [i for i in self._outputs if i in nlzd_idcs]
        new_tf = self.__class__(
            m=new_m,
            outputs=new_outputs,
            batch_shape=self._batch_shape,
            min_stdv=self._min_stdv,
        )
        if self.means is not None:
            new_tf.means = self.means[..., nlzd_idcs]
            new_tf.stdvs = self.stdvs[..., nlzd_idcs]
            new_tf._stdvs_sq = self._stdvs_sq[..., nlzd_idcs]
        if not self.training:
            new_tf.eval()
        return new_tf

    def untransform(
        self, Y: Tensor, Yvar: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Un-standardize outcomes.

        Args:
            Y: A `batch_shape x n x m`-dim tensor of standardized targets.
            Yvar: A `batch_shape x n x m`-dim tensor of standardized observation
                noises associated with the targets (if applicable).

        Returns:
            A two-tuple with the un-standardized outcomes:

            - The un-standardized outcome observations.
            - The un-standardized observation noise (if applicable).
        """
        if self.means is None:
            raise RuntimeError(
                "`Standardize` transforms must be called on outcome data "
                "(e.g. `transform(Y)`) before calling `untransform`, since "
                "means and standard deviations need to be computed."
            )

        Y_utf = self.means + self.stdvs * Y
        Yvar_utf = self._stdvs_sq * Yvar if Yvar is not None else None
        return Y_utf, Yvar_utf

    @property
    def _is_linear(self) -> bool:
        return True

    def untransform_posterior(
        self, posterior: Posterior
    ) -> Union[GPyTorchPosterior, TransformedPosterior]:
        r"""Un-standardize the posterior.

        Args:
            posterior: A posterior in the standardized space.

        Returns:
            The un-standardized posterior. If the input posterior is a
            `GPyTorchPosterior`, return a `GPyTorchPosterior`. Otherwise, return a
            `TransformedPosterior`.
        """
        if self._outputs is not None:
            raise NotImplementedError(
                "Standardize does not yet support output selection for "
                "untransform_posterior"
            )
        if self.means is None:
            raise RuntimeError(
                "`Standardize` transforms must be called on outcome data "
                "(e.g. `transform(Y)`) before calling `untransform_posterior`, since "
                "means and standard deviations need to be computed."
            )
        is_mtgp_posterior = False
        if type(posterior) is GPyTorchPosterior:
            is_mtgp_posterior = posterior._is_mt
        if not self._m == posterior._extended_shape()[-1] and not is_mtgp_posterior:
            raise RuntimeError(
                "Incompatible output dimensions encountered. Transform has output "
                f"dimension {self._m} and posterior has "
                f"{posterior._extended_shape()[-1]}."
            )

        if type(posterior) is not GPyTorchPosterior:
            # fall back to TransformedPosterior
            # this applies to subclasses of GPyTorchPosterior like MultitaskGPPosterior
            return TransformedPosterior(
                posterior=posterior,
                sample_transform=lambda s: self.means + self.stdvs * s,
                mean_transform=lambda m, v: self.means + self.stdvs * m,
                variance_transform=lambda m, v: self._stdvs_sq * v,
            )
        # GPyTorchPosterior (TODO: Should we Lazy-evaluate the mean here as well?)
        mvn = posterior.distribution
        offset = self.means
        scale_fac = self.stdvs
        if not posterior._is_mt:
            mean_tf = offset.squeeze(-1) + scale_fac.squeeze(-1) * mvn.mean
            scale_fac = scale_fac.squeeze(-1).expand_as(mean_tf)
        else:
            mean_tf = offset + scale_fac * mvn.mean
            reps = mean_tf.shape[-2:].numel() // scale_fac.size(-1)
            scale_fac = scale_fac.squeeze(-2)
            if mvn._interleaved:
                scale_fac = scale_fac.repeat(*[1 for _ in scale_fac.shape[:-1]], reps)
            else:
                scale_fac = torch.repeat_interleave(scale_fac, reps, dim=-1)

        if (
            not mvn.islazy
            # TODO: Figure out attribute namming weirdness here
            or mvn._MultivariateNormal__unbroadcasted_scale_tril is not None
        ):
            # if already computed, we can save a lot of time using scale_tril
            covar_tf = CholLinearOperator(mvn.scale_tril * scale_fac.unsqueeze(-1))
        else:
            lcv = mvn.lazy_covariance_matrix
            scale_fac = scale_fac.expand(lcv.shape[:-1])
            scale_mat = DiagLinearOperator(scale_fac)
            covar_tf = scale_mat @ lcv @ scale_mat

        kwargs = {"interleaved": mvn._interleaved} if posterior._is_mt else {}
        mvn_tf = mvn.__class__(mean=mean_tf, covariance_matrix=covar_tf, **kwargs)
        return GPyTorchPosterior(mvn_tf)
