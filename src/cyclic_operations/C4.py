"""
Implementation of the C4 Equivariant CNN framework from 

'Exploiting Cyclic Symmetry in Convolutional Neural Networks'
by Sander Dieleman, Jeffrey De Fauw, Koray Kavukcuoglu

The C4 Symmetry group consists of k*90 degree rotations, for k
in {0,1,2,3}. The framework is made equivariant to these rotations
by exhausting all transformations in the group through four operations:

- Slice; Given an input X, create 4 paths (one for each rotation).
- Stack; Given a sliced minibatch of feature maps, undo the slicing,
    by rotating all feature maps (of the same sample) to the 
    same orientation.
- Roll; Perform 4 Stack operations, to quadruple the number of feature
    maps for each path. To get different orientations out of the stack
    operation, the Cyclic Permutation (sigma in the paper) is used.
- Pool; After sufficient convolutions, the 4 paths are pooled back to 
    their original orientation, with a permutation invariant pooling
    function.

Author:
- Alec Noppe @alecnoppe
"""

import torch
from torch import nn


class C4_Cyclic_Permutation(nn.Module):
    def __init__(self):
        """
        Cyclic Permutation (backwards shift);
        Refered to as 'sigma' in the paper
        """
        super().__init__()

    def forward(self, x, k=1):
        """
        The Cyclic Permutation operation shifts/rolls all orientations of the sample
        backwards.
        
        If we have one sample [x] that is sliced to become [x1, x2, x3, x4],
        we can perform a single cyclic permutation to get

        [x1, x2, x3, x4] -> [x2, x3, x4, x1]

        Args:
            x: torch.tensor; Mini-batch of sliced samples
            k: int; Number of shifts/rolls to perform.
        
        Returns:
            permuted_x: torch.tensor; k-Shifted Mini-batch
        """
        assert len(x.shape) == 4, "Tensor should have dimensions (B,C,H,W)"
        # Transform the mini-batch such that we can shift per sample
        # (By creating a temporary dimension for all orientations (dim 1))
        B,C,H,W = x.shape
        permuted_x = x.view(B//4, 4, C, H, W)
        # Roll (backwards shift) the mini-batch, such that all elements
        # are shifted to the left by k.
        permuted_x = torch.roll(permuted_x, -1*k, dims=1)
        # Finally, zip/interleave the samples back in their original order
        permuted_x = permuted_x.flatten(start_dim=0, end_dim=1)
        return permuted_x


class C4_Slice(nn.Module):
    def __init__(self):
        """C4-Slice operation from the paper."""
        super().__init__()

    def forward(self, x):
        """
        The C4 Slice operation takes the input mini-batch,
        and rotates each sample with k*90 degrees for k in {0,1,2,3}.
        
        S(x) = [x, rx, rrx, rrrx]
        where S(x) is the slice operation, and r is a 90 degree rotation.

        Afterwards, these samples are stacked along the batch dimension,
        quadrupling the batch-size.

        Args:
            x: torch.tensor; un-sliced input samples.

        Returns:
            X_sliced: torch.tensor; C4-Sliced tensor
        """
        assert len(x.shape) == 4, "Tensor should have dimensions (B,C,H,W)"
        # Create all k*90 degree rotations for k in {0,1,2,3}
        x_90  = torch.rot90(x, -1, dims=(2,3))
        x_180 = torch.rot90(x, -2, dims=(2,3))
        x_270 = torch.rot90(x, -3, dims=(2,3))
        # Stack the rotated paths on-top of eachother
        X_sliced = torch.stack((x, x_90, x_180, x_270), dim=1)
        # Finally, zip/interleave the samples such that rotations of the same
        # sample are adjacent.
        X_sliced = torch.flatten(X_sliced, start_dim=0, end_dim=1)

        return (sliced_x := X_sliced)


class C4_Pool(nn.Module):
    def __init__(self, pooling="mean"):
        """
        C4-Pool operation from the paper.
        
        Args:
            pooling:str "mean" or "sum"; Which pooling function to use
        """
        super().__init__()
        self.pooling_fn = self._mean_pool if pooling == "mean" else self._sum_pool

    def _mean_pool(self, unrotated_x):
        return unrotated_x.mean(dim=1)

    def _sum_pool(self, unrotated_x):
        return unrotated_x.sum(dim=1)

    def forward(self, x):
        """
        The C4-Pool operation takes a mini-batch of rotated/sliced feature maps,
        and uses a permutation invariant pooling function to combine the 
        separate paths (orientations of each sample).

        This function preserves equivariance, since the separate paths
        are rotated in the original direction prior to pooling.

        P(x) = p([x, r'x, r'r'x, r'r'r'x])
        where p() is the permutation invariant pooling function,
        r' indicates a 90 degree rotation in the inverse direction.

        Reduces batch-size by 4.

        Args:
            x: torch.tensor; sliced mini-batch

        Returns:
            x: torch.tensor; un-sliced and pooled mini-batch
        """
        # Create a temporary dimension (dim=1) such that we can
        # Un-rotate the paths of each sample, without messing
        # with the order of the samples.
        B,C,H,W = x.shape
        unfolded_x = x.view(B//4, 4, C, H, W)
        # Un-rotate the samples.
        unfolded_x[:, 1, ...] = torch.rot90(unfolded_x[:, 1, ...], 1, dims=(2,3))
        unfolded_x[:, 2, ...] = torch.rot90(unfolded_x[:, 2, ...], 2, dims=(2,3))
        unfolded_x[:, 3, ...] = torch.rot90(unfolded_x[:, 3, ...], 3, dims=(2,3))
        # Finally, pool the different paths using a permutation-invariant 
        # pooling function.
        return (pooled_x := self.pooling_fn(unfolded_x))


class C4_Stack(nn.Module):
    def __init__(self):
        """
        C4-Stack operation from the paper
        """
        super().__init__()

    def forward(self, x):
        """
        The C4-Stack operation takes a mini-batch of rotated/sliced feature maps,
        and undoes the rotations of each sample. The un-rotated samples are then
        stacked along one batch dimension (reducing the batch-size fourfold, and
        quadrupling the number of feature maps).

        T(x) = [x, r'x, r'r'x, r'r'r'x]
        where T(x) is the stack operation, and
        r' indicates a 90 degree rotation in the inverse direction.

        Args:
            x: torch.tensor; sliced mini-batch

        Returns:
            x: torch.tensor; un-sliced and unrotated mini-batch
        """
        # Create a temporary dimension (dim=1) such that we can
        # Un-rotate the paths of each sample, without messing
        # with the order of the samples.
        B,C,H,W = x.shape
        unfolded_x = x.view(B//4, 4, C, H, W)
        # Un-rotate the feature maps.
        unfolded_x[:, 1, ...] = torch.rot90(unfolded_x[:, 1, ...], 1, dims=(2,3))
        unfolded_x[:, 2, ...] = torch.rot90(unfolded_x[:, 2, ...], 2, dims=(2,3))
        unfolded_x[:, 3, ...] = torch.rot90(unfolded_x[:, 3, ...], 3, dims=(2,3))
        # Stack the feature maps along the original batch axes,
        # Thus reducing the batch-size.
        return (stacked_x := unfolded_x.view(B//4, C*4, H, W))


class C4_Roll(nn.Module):
    def __init__(self, use_version2 = False):
        """
        C4-Roll operation from the paper.

        Args:
            use_version2: bool; whether to re-align the feature maps after original
                roll operation.
                NOTE: Does not show any advantage
        """
        super().__init__()
        # Use the cyclic permutation and C4-Stack operations
        self.permute = C4_Cyclic_Permutation()
        self.stack = C4_Stack()
        self.use_version2 = use_version2

    def forward(self, x):
        """
        The C4-Roll operation uses the Cyclic Permutation (backwards shift)
        and C4-Stack to effectively quadruple the feature maps, without
        any additional model parameters. This is achieved by un-rotating the
        feature maps for each sample, and stacking them along the
        feature map dimension.

        R(x) = [T(x), T(sx), T(ssx), T(sssx)]
        where R(x) is the C4-Roll operation,
        T(x) is the C4-Stack operation,
        and each s indicates a backwards shift with the Cyclic Permutation
        operation.

        Args:
            x: torch.tensor; sliced mini-batch

        Returns:
            x: torch.tensor; richer mini-batch
        """
        # Permute and stack the feature maps along each path.
        B,C,H,W = x.shape
        X_roll_tuple = (self.stack(torch.clone(x)), 
                        self.stack(self.permute(torch.clone(x), 1)), 
                        self.stack(self.permute(torch.clone(x), 2)), 
                        self.stack(self.permute(torch.clone(x), 3)))
        # Version2 of the Roll operation re-aligns the channels to have the same
        # Activations on each dimension
        if self.use_version2:
            X_roll_tuple = X_roll_tuple[0], self.permute(X_roll_tuple[1], -1), self.permute(X_roll_tuple[2], -2), self.permute(X_roll_tuple[3], -3)

        # Re-structure the tensor to the sliced batch dimensions
        X_roll = torch.stack(X_roll_tuple, dim=1).view(B,C*4,H,W)

        return (rolled_x:= X_roll)
