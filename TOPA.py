import torch
import torch.nn as nn

def matrix_sqrt(mat, epsilon=1e-6):
    """
    Compute the square root and inverse square root of a symmetric positive-definite matrix,
    with numerical stability considerations.

    Args:
        mat (torch.Tensor): Symmetric positive-definite matrix of shape [N, N].
        epsilon (float): Small value to ensure numerical stability.

    Returns:
        torch.Tensor: Square root of the matrix.
        torch.Tensor: Inverse square root of the matrix.
    """
    # Perform eigenvalue decomposition for the symmetric matrix
    eigvals, eigvecs = torch.linalg.eigh(mat)
    # Clamp eigenvalues to ensure numerical stability
    eigvals = torch.clamp(eigvals, min=epsilon)
    # Compute square root and inverse square root of eigenvalues
    eigvals_sqrt = torch.sqrt(eigvals)
    eigvals_inv_sqrt = 1.0 / eigvals_sqrt

    # Reconstruct the square root and inverse square root matrices
    sqrt_mat = eigvecs @ torch.diag(eigvals_sqrt) @ eigvecs.T
    inv_sqrt_mat = eigvecs @ torch.diag(eigvals_inv_sqrt) @ eigvecs.T
    return sqrt_mat, inv_sqrt_mat

def Entropy(input_):
    """
    Compute the entropy for a given input tensor.

    Args:
        input_ (torch.Tensor): Input tensor of shape [B, C], where B is the batch size and C is the number of classes.

    Returns:
        torch.Tensor: Entropy values of shape [B].
    """
    bs = input_.size(0)
    epsilon = 1e-5  # Small value to avoid log(0)
    # Compute entropy using the formula -p * log(p)
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def riemannian_distance(A, B, epsilon=1e-6):
    """
    Compute the Riemannian distance between two symmetric positive-definite matrices.

    Args:
        A (torch.Tensor): Symmetric positive-definite matrix of shape [N, N].
        B (torch.Tensor): Symmetric positive-definite matrix of shape [N, N].
        epsilon (float): Small value to ensure numerical stability.

    Returns:
        torch.Tensor: Riemannian distance between A and B.
    """
    # Compute A^(-1/2)
    _, A_inv_sqrt = matrix_sqrt(A, epsilon)

    # Compute A^(-1/2) * B * A^(-1/2)
    C = A_inv_sqrt @ B @ A_inv_sqrt

    # Perform eigenvalue decomposition for the intermediate matrix
    eigvals, eigvecs = torch.linalg.eigh(C)
    # Clamp eigenvalues to ensure numerical stability
    eigvals = torch.clamp(eigvals, min=epsilon)
    # Compute the matrix logarithm
    log_C = eigvecs @ torch.diag(torch.log(eigvals)) @ eigvecs.T

    # Compute the Frobenius norm of the logarithm matrix
    distance = torch.norm(log_C, p='fro')
    return distance

def normalize_correlation_matrix(G, epsilon=1e-8):
    """
    Normalize a correlation matrix G (e.g., Gram matrix).

    Args:
        G (torch.Tensor): Input correlation matrix of shape [C, C].
        epsilon (float): Small value to avoid division by zero.

    Returns:
        torch.Tensor: Normalized correlation matrix of shape [C, C].
    """
    # Ensure G is a square matrix
    assert G.shape[0] == G.shape[1], "Input matrix G must be square."
    # Compute row and column sums of G
    row_sum = torch.sum(G, dim=1, keepdim=True)
    col_sum = torch.sum(G, dim=0, keepdim=True)
    # Compute normalization factors using outer product
    norm_factors = torch.sqrt(row_sum @ col_sum + epsilon)
    # Normalize the correlation matrix
    G_normalized = G / norm_factors
    return G_normalized


def TOPA_Loss(outputs_target, prototypes, cls_normalization=False,proto_detach = False,t = 2.5 ):
    """
    Compute the TOPA loss based on target outputs and prototypes.

    Args:
        outputs_target (torch.Tensor): Model predictions for the target domain of shape [B, C].
        prototypes (torch.Tensor): Class prototypes of shape [D, C], where D is the feature dimension.
        cls_normalization (bool): Whether to normalize the correlation matrices.

    Returns:
        torch.Tensor: TOPA loss value.
    """
     # Temperature scaling factor
    class_num = prototypes.size(0)  # Number of classes

    # Apply temperature scaling to target outputs
    outputs_target_temp = outputs_target / t
    # Compute softmax probabilities
    target_softmax_out_temp = nn.Softmax(dim=1)(outputs_target_temp)

    # Compute entropy-based weights for target domain samples
    target_entropy_weight = Entropy(target_softmax_out_temp).detach()
    target_entropy_weight = 1 + torch.exp(-target_entropy_weight)
    target_entropy_weight = outputs_target.size(0) * target_entropy_weight / torch.sum(target_entropy_weight)

    # Compute the weighted covariance matrix for the target domain
    cov_matrix_t = target_softmax_out_temp.mul(target_entropy_weight.view(-1, 1)).transpose(1, 0).mm(target_softmax_out_temp)

    # Optionally normalize the target covariance matrix
    if cls_normalization:
        cov_matrix_t = normalize_correlation_matrix(cov_matrix_t)
    else:
        cov_matrix_t = cov_matrix_t * class_num / outputs_target.shape[0]

    # Compute the pairwise distance between prototypes
    if proto_detach:
        y_p = -torch.cdist(prototypes.detach(), prototypes.detach(), p=2)
    else:
        y_p = -torch.cdist(prototypes, prototypes, p=2)

    # Apply temperature scaling to prototypes
    outputs_p_temp = y_p / t
    # Compute softmax probabilities for prototypes
    p_softmax_out_temp = nn.Softmax(dim=1)(outputs_p_temp)

    # Compute the covariance matrix for prototypes
    cov_matrix_p = p_softmax_out_temp.transpose(1, 0).mm(p_softmax_out_temp)

    # Optionally normalize the prototype covariance matrix
    if cls_normalization:
        cov_matrix_p = normalize_correlation_matrix(cov_matrix_p)

    # Compute the Riemannian distance between the two covariance matrices
    topa_loss = riemannian_distance(cov_matrix_t, cov_matrix_p)
    return topa_loss
