"""
Extension of the C4 Equivariant CNN to a D4 Equivariant CNN,
by including reflections in the transformation group.

The C4 Equivariant CNN is from
'Exploiting Cyclic Symmetry in Convolutional Neural Networks'
by Sander Dieleman, Jeffrey De Fauw, Koray Kavukcuoglu
Extensions are my own.

The D4 Symmetry group consists of k*90 degree rotations, for k
in {0,1,2,3} and the reflections of these transformations. 
The framework is made equivariant to these rotations
by exhausting all transformations in the group through four operations:

- Slice; Given an input X, create 8 paths (one for each rotation and all reflections).
- Stack; Given a sliced minibatch of feature maps, undo the slicing,
    by rotating/reflecting all feature maps (of the same sample) to the 
    same orientation.
- Roll; Perform 8 Stack operations, to increase the number of feature
    maps for each path with factor 8. To get different orientations out of the stack
    operation, the Cyclic Permutation is used.
- Pool; After sufficient convolutions, the 8 paths are pooled back to 
    their original orientation, with a permutation invariant pooling
    function.

Author:
- Alec Noppe @alecnoppe
"""

import torch
from torch import nn


class D4_Cyclic_Permutation(nn.Module):
    def __init__(self):
        """
        Cyclic Permutation (backwards shift);
        Modification of the one outlined in the paper, to work for
        non commuting transformations (ie D4 group).
        """
        super().__init__()

    def forward(self, x, k=1):
        """
        The Cyclic Permutation operation shifts/rolls all orientations of the sample
        backwards - separated for the rotations and the reflected rotations.
        
        If we have one sample [x] that is sliced to become 
        [x1, x2, x3, x4, Rx1, Rx2, Rx3, Rx4]
        where x1 - x4 are the C4 group and Rx1 - RX4 are the horizontal reflections
        of the corresponding C4 orientation.

        The cyclic shift here has to account for the non-commutative property of the
        transformations. This is achieved by shifting drawing an imaginary line between
        the C4 group and the reflections, and shifting within the two sections separately

        ie for one backwards shift using this algorithm, we get:
        [x1, x2, x3, x4, Rx1, Rx2, Rx3, Rx4]
                        |
                        V
        [x2, x3, x4, x1,  Rx2, Rx3, Rx4, Rx1]

        Here we shift the C4 group backwards within the bounds of the C4 paths
        and the D4 group backwards within the bounds of the D4 paths.


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
        permuted_x = x.view(B//8, 8, C, H, W)
        # Roll (backwards shift) the C4 paths, and the D4 paths separately
        # (See the :4 and 4: indexing)
        permuted_x[:, :4, ...] = torch.roll(permuted_x[:, :4, ...], -1*k, dims=1)
        permuted_x[:, 4:, ...] =  torch.roll(permuted_x[:, 4:, ...], -1*k, dims=1)
        permuted_x = x.view(B//8, 8, C, H, W)
        # Finally, zip/interleave the samples back in their original order
        permuted_x = permuted_x.flatten(start_dim=0, end_dim=1)
        return permuted_x


class D4_Slice(nn.Module):
    def __init__(self):
        """D4-Slice operation."""
        super().__init__()

    def forward(self, x):
        """
        The D4 Slice operation takes the input mini-batch,
        and rotates each sample with k*90 degrees for k in {0,1,2,3}
        and their horizontal reflections.
        
        S(x) = [x, rx, rrx, rrrx, Rx, Rrx, Rrrx, Rrrrx]
        where S(x) is the slice operation, and r is a 90 degree rotation,
        and R is a horizontal reflection.

        Afterwards, these samples are stacked along the batch dimension,
        increasing the batch-size by factor 8.

        Args:
            x: torch.tensor; un-sliced input samples.

        Returns:
            X_sliced: torch.tensor; C4-Sliced tensor
        """
        assert len(x.shape) == 4, "Tensor should have dimensions (B,C,H,W)"
        # Create all k*90 degree rotations for k in {0,1,2,3}
        x_90   = torch.rot90(x, -1, dims=(2,3))
        x_180  = torch.rot90(x, -2, dims=(2,3))
        x_270  = torch.rot90(x, -3, dims=(2,3))
        # Create all reflections fo the k*90 degree rotations
        xr     = torch.flip(x, dims=(3, ))
        xr_90  = torch.flip(x_90, dims=(3, ))
        xr_180 = torch.flip(x_180, dims=(3, ))
        xr_270 = torch.flip(x_270, dims=(3, ))
        # Stack the rotated paths on-top of eachother
        X_sliced = torch.stack((x, x_90, x_180, x_270,
                                xr, xr_90, xr_180, xr_270), 
                                dim=1)
        # Finally, zip/interleave the samples such that orientations of the same
        # sample are adjacent.
        X_sliced = torch.flatten(X_sliced, start_dim=0, end_dim=1)

        return (sliced_x := X_sliced)


class D4_Pool(nn.Module):
    def __init__(self, pooling="mean"):
        """
        D4-Pool operation.
        
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
        The D4-Pool operation takes a mini-batch of rotated/reflected feature maps,
        and uses a permutation invariant pooling function to combine the 
        separate paths (orientations of each sample).

        This function preserves equivariance, since the separate paths
        are rotated/reflected in the original direction prior to pooling.

        P(x) = p([x, r'x, r'r'x, r'r'r'x, Rx, Rr'x, Rr'r'x, Rr'r'r'x])
        where p() is the permutation invariant pooling function,
        r' indicates a 90 degree rotation in the inverse direction,
        R is a reflection.

        Reduces batch-size by 8.

        Args:
            x: torch.tensor; sliced mini-batch

        Returns:
            x: torch.tensor; un-sliced and pooled mini-batch
        """
        # Create a temporary dimension (dim=1) such that we can
        # Un-rotate/reflect the paths of each sample, without messing
        # with the order of the samples.
        B,C,H,W = x.shape
        unfolded_x = x.view(B//8, 8, C, H, W)
        # Un-rotate the samples.
        unfolded_x[:, 1, ...] = torch.rot90(unfolded_x[:, 1, ...], 1, dims=(2,3))
        unfolded_x[:, 2, ...] = torch.rot90(unfolded_x[:, 2, ...], 2, dims=(2,3))
        unfolded_x[:, 3, ...] = torch.rot90(unfolded_x[:, 3, ...], 3, dims=(2,3))
        # Reflect, then un-rotate the reflected paths
        unfolded_x[:, 4, ...] = torch.flip(unfolded_x[:, 4, ...], dims=(3, ))
        unfolded_x[:, 5, ...] = torch.rot90(torch.flip(unfolded_x[:, 5, ...], dims=(3, )), 1, dims=(2,3))
        unfolded_x[:, 6, ...] = torch.rot90(torch.flip(unfolded_x[:, 6, ...], dims=(3, )), 2, dims=(2,3))
        unfolded_x[:, 7, ...] = torch.rot90(torch.flip(unfolded_x[:, 7, ...], dims=(3, )), 3, dims=(2,3))
        # Finally, pool the different paths using a permutation-invariant 
        # pooling function.
        return (pooled_x := self.pooling_fn(unfolded_x))


class D4_Stack(nn.Module):
    def __init__(self):
        """
        D4-Stack operation.
        """
        super().__init__()

    def forward(self, x, version1=True):
        """
        The D4-Stack operation takes a mini-batch of rotated/reflected feature maps,
        and undoes the rotations of each sample. The un-rotated/un-reflected samples 
        are then stacked along one batch dimension.

        Effectively reducing the batch-size by factor 8, and increasing the number of
        feature maps by factor 8.

        Since reflections and rotations are not commutative, we make two variants of the
        stack algorithm (version1=True/False).

        We use Version1, to create the stacks for the C4 group orientations (so 
        rotations without reflections) as follows:

        T(x) = [x, r'x, r'r'x, r'r'r'x, Rx, r'Rx, r'r'Rx, r'r'r'Rx]
        where T(x) is the D4-Stack function,
        r' indicates a 90 degree rotation in the inverse direction,
        R is a reflection.

        We use Version2 to create the stacks for the reflected C4 group orientations
        (so all orientations WITH reflections) as follows:

        T(x) = [Rx, Rr'x, Rr'r'x, Rr'r'r'x, x, rx, rrx, rrrx]
        where T(x) is the D4-Stack function,
        r indicates a 90 degree rotation,
        r' indicates a 90 degree rotation in the inverse direction
        R is a reflection.

        Args:
            x: torch.tensor; sliced mini-batch

        Returns:
            x: torch.tensor; un-sliced and unrotated mini-batch
        """
        # Create a temporary dimension (dim=1) such that we can
        # Un-rotate/reflect the paths of each sample, without messing
        # with the order of the samples.
        B,C,H,W = x.shape
        unfolded_x = x.view(B//8, 8, C, H, W)
        # Version1; create same-orientations for the C4 paths
        if version1:
            # Un-rotate the C4 feature maps.
            unfolded_x[:, 1, ...] = torch.rot90(unfolded_x[:, 1, ...], 1, dims=(2,3))
            unfolded_x[:, 2, ...] = torch.rot90(unfolded_x[:, 2, ...], 2, dims=(2,3))
            unfolded_x[:, 3, ...] = torch.rot90(unfolded_x[:, 3, ...], 3, dims=(2,3))
            # Reflect, then un-rotate the reflected C4 feature maps.
            unfolded_x[:, 4, ...] = torch.flip(unfolded_x[:, 4, ...], dims=(3, ))
            unfolded_x[:, 5, ...] = torch.rot90(torch.flip(unfolded_x[:, 5, ...], dims=(3, )), 1, dims=(2,3))
            unfolded_x[:, 6, ...] = torch.rot90(torch.flip(unfolded_x[:, 6, ...], dims=(3, )), 2, dims=(2,3))
            unfolded_x[:, 7, ...] = torch.rot90(torch.flip(unfolded_x[:, 7, ...], dims=(3, )), 3, dims=(2,3))
            # Stack the feature maps along the original batch axes,
            # Thus reducing the batch-size.
            return (stacked_x := unfolded_x.view(B//8, C*8, H, W))
        # Version2; create same-orientations for the reflected C4 paths
        # Un-rotate then reflect C4 feature maps.
        unfolded_x[:, 0, ...] = torch.flip(unfolded_x[:, 0, ...], dims=(3, ))
        unfolded_x[:, 1, ...] = torch.flip(torch.rot90(unfolded_x[:, 1, ...], 1, dims=(2,3)), dims=(3, ))
        unfolded_x[:, 2, ...] = torch.flip(torch.rot90(unfolded_x[:, 2, ...], 2, dims=(2,3)), dims=(3, ))
        unfolded_x[:, 3, ...] = torch.flip(torch.rot90(unfolded_x[:, 3, ...], 3, dims=(2,3)), dims=(3, ))
        # Rotate the reflected feature maps
        unfolded_x[:, 5, ...] = torch.rot90(unfolded_x[:, 5, ...], -1, dims=(2,3))
        unfolded_x[:, 6, ...] = torch.rot90(unfolded_x[:, 6, ...], -2, dims=(2,3))
        unfolded_x[:, 7, ...] = torch.rot90(unfolded_x[:, 7, ...], -3, dims=(2,3))
        # Stack the feature maps along the original batch axes,
        # Thus reducing the batch-size.
        return (stacked_x := unfolded_x.view(B//8, C*8, H, W))


class D4_Roll(nn.Module):
    def __init__(self, use_version2 = False):
        """
        D4-Roll operation from the paper.

        Args:
            use_version2: bool; whether to re-align the feature maps after original
                roll operation.
                NOTE: Does not show any advantage
        """
        super().__init__()
        # Use the cyclic permutation and D4-Stack operations
        self.permute = D4_Cyclic_Permutation()
        self.stack = D4_Stack()
        self.use_version2 = use_version2

    def forward(self, x):
        """
        The D4-Roll operation uses the Cyclic Permutation (backwards shift)
        and D4-Stack to effectively increase the feature maps by factor 8, without
        any additional model parameters. This is achieved by un-rotating/reflecting
        the feature maps for each sample, and stacking them along the
        feature map dimension.

        R(x) = [T_v1(x), T_v1(sx), T_v1(ssx), T_v1(sssx), 
                T_v2(x), T_v2(sx), T_v2(ssx), T_v2(sssx)]
        where R(x) is the D4-Roll operation,
        T_v1(x) is the D4-Stack operation (version1),
        T_v2(x) is the D4-Stack operation (version2),
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
                        self.stack(self.permute(torch.clone(x), 3)),
                        self.stack(torch.clone(x), version1=False), 
                        self.stack(self.permute(torch.clone(x), 1), version1=False), 
                        self.stack(self.permute(torch.clone(x), 2), version1=False), 
                        self.stack(self.permute(torch.clone(x), 3), version1=False),
                        )
        
        # Version2 of the Roll operation re-aligns the channels to have the same
        # Activations on each dimension
        if self.use_version2:
            X_roll_tuple = X_roll_tuple[0], self.permute(X_roll_tuple[1], -1), self.permute(X_roll_tuple[2], -2), self.permute(X_roll_tuple[3], -3), \
                X_roll_tuple[4], self.permute(X_roll_tuple[5], -1), self.permute(X_roll_tuple[6], -2), self.permute(X_roll_tuple[7], -3)

        # Re-structure the tensor to the sliced batch dimensions
        X_roll = torch.stack(X_roll_tuple, dim=1).view(B,C*8,H,W)

        return (rolled_x:= X_roll)
    