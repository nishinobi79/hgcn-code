import torch
import torch.nn.functional as F
import numpy as np

PROJ_EPS = 1e-5
EPS = 1e-15
MAX_TANH_ARG = 15.0
 

def torch_project_hyp_vecs(x, c):
    # Projection op. Need to make sure hyperbolic embeddings are inside the unit ball.
    return torch.nn.functional.normalize(x, p=2, dim=1, eps=PROJ_EPS / np.sqrt(c))


######################## x, y have shape [batch_size, emb_dim] in all torch_* functions ################

# Real x, not vector!
def torch_atanh(x):
    return torch.atanh(torch.minimum(x, 1. - EPS))  # Only works for positive real x.


# Real x, not vector!
def torch_tanh(x):
    return torch.tanh(torch.minimum(torch.maximum(x, -MAX_TANH_ARG), MAX_TANH_ARG))


def torch_dot(x, y):
    return torch.sum(x * y, dim=1, keepdim=True)


def torch_norm(x):
    return torch.norm(x, dim=1, keepdim=True)


#########################
def torch_mob_add(u, v, c):
    v = v + EPS
    torch_dot_u_v = 2. * c * torch_dot(u, v)
    torch_norm_u_sq = c * torch_dot(u, u)
    torch_norm_v_sq = c * torch_dot(v, v)
    denominator = 1. + torch_dot_u_v + torch_norm_v_sq * torch_norm_u_sq
    result = (1. + torch_dot_u_v + torch_norm_v_sq) / denominator * u + (1. - torch_norm_u_sq) / denominator * v
    return torch_project_hyp_vecs(result, c)


#########################
def torch_poinc_dist_sq(u, v, c):
    sqrt_c = np.sqrt(c)
    m = torch_mob_add(-u, v, c) + EPS
    atanh_x = np.sqrt(c) * torch_norm(m)
    dist_poincare = 2. / sqrt_c * torch_atanh(atanh_x)
    return dist_poincare ** 2


#########################
def torch_euclid_dist_sq(u, v):
    return torch.sum((u - v) ** 2, dim=1, keepdim=True)


#########################
def torch_mob_scalar_mul(r, v, c):
    v = v + EPS
    norm_v = torch_norm(v)
    nomin = torch_tanh(r * torch_atanh(np.sqrt(c) * norm_v))
    result = nomin / (np.sqrt(c) * norm_v) * v
    return torch_project_hyp_vecs(result, c)


#########################
def torch_lambda_x(x, c):
    return 2. / (1 - c * torch_dot(x, x))


#########################
def unit_speed_geo(x, v, t, c):
    second_term = np.tanh(np.sqrt(c) * t / 2) / (np.sqrt(c) * torch_norm(v)) * v
    return torch_mob_add(x, second_term, c)


def exp_map_x(x, v, c):
    second_term = np.tanh(np.sqrt(c) * torch_lambda_x(x, c) * torch_norm(v) / 2) / (
            np.sqrt(c) * torch_norm(v)) * v
    return torch_mob_add(x, second_term, c)


def log_map_x(x, y, c):
    diff = torch_mob_add(-x, y, c)
    lam = torch_lambda_x(x, c)
    return 2. / (np.sqrt(c) * lam) * torch_atanh(np.sqrt(c) * torch_norm(diff)) / torch_norm(
        diff) * diff


def torch_exp_map_x(x, v, c):
    v = v + EPS  # Perturbe v to avoid dealing with v = 0
    norm_v = torch_norm(v)
    second_term = (torch_tanh(np.sqrt(c) * torch_lambda_x(x, c) * norm_v / 2) / (
            np.sqrt(c) * norm_v)) * v
    return torch_mob_add(x, second_term, c)


def torch_log_map_x(x, y, c):
    diff = torch_mob_add(-x, y, c) + EPS
    norm_diff = torch_norm(diff)
    lam = torch_lambda_x(x, c)
    return (((2. / np.sqrt(c)) / lam) * torch_atanh(np.sqrt(c) * norm_diff) / norm_diff) * diff


def torch_exp_map_zero(v, c):
    v = v + EPS  # Perturbe v to avoid dealing with v = 0
    norm_v = torch_norm(v)
    result = torch_tanh(np.sqrt(c) * norm_v) / (np.sqrt(c) * norm_v) * v
    return torch_project_hyp_vecs(result, c)


def torch_log_map_zero(y, c):
    diff = y + EPS
    norm_diff = torch_norm(diff)
    return 1. / np.sqrt(c) * torch_atanh(np.sqrt(c) * norm_diff) / norm_diff * diff


#########################
def torch_mob_mat_mul(M, x, c):
    x = x + EPS
    Mx = torch.matmul(x, M) + EPS
    MX_norm = torch_norm(Mx)
    x_norm = torch_norm(x)
    result = 1. / np.sqrt(c) * torch_tanh(
        MX_norm / x_norm * torch_atanh(np.sqrt(c) * x_norm)) / MX_norm * Mx
    return torch_project_hyp_vecs(result, c)


# x is hyperbolic, u is Euclidean. Computes diag(u) times x.
def torch_mob_pointwise_prod(x, u, c):
    x = x + EPS
    Mx = x * u + EPS
    MX_norm = torch_norm(Mx)
    x_norm = torch_norm(x)
    result = 1. / np.sqrt(c) * torch_tanh(
        MX_norm / x_norm * torch_atanh(np.sqrt(c) * x_norm)) / MX_norm * Mx
    return torch_project_hyp_vecs(result, c)


#########################
def riemannian_gradient_c(u, c):
    return ((1. - c * torch_dot(u, u)) ** 2) / 4.0


#########################
def torch_eucl_non_lin(eucl_h, non_lin):
    if non_lin == 'id':
        return eucl_h
    elif non_lin == 'relu':
        return F.relu(eucl_h)
    elif non_lin == 'tanh':
        return torch.tanh(eucl_h)
    elif non_lin == 'sigmoid':
        return torch.sigmoid(eucl_h)
    return eucl_h


# Applies a non linearity sigma to a hyperbolic h using: exp_0(sigma(log_0(h)))
def torch_hyp_non_lin(hyp_h, non_lin, hyp_output, c):
    if non_lin == 'id':
        if hyp_output:
            return hyp_h
        else:
            return torch_log_map_zero(hyp_h, c)

    eucl_h = torch_eucl_non_lin(torch_log_map_zero(hyp_h, c), non_lin)

    if hyp_output:
        return torch_exp_map_zero(eucl_h, c)
    else:
        return eucl_h


####################################################################################################
####################################################################################################
####################################### Unit tests #################################################
####################################################################################################
####################################################################################################
def mobius_addition_left_cancelation_test():
    for i in range(0, 10000):
        a = np.random.uniform(low=-0.01, high=0.01, size=1)
        b = np.random.uniform(low=-0.01, high=0.01, size=1)

        c = random.random()
        res = torch_mob_add(-a, torch_mob_add(a, b, c=c), c=c)
        diff = np.sum(np.abs(b - res.cpu().numpy()))
        if diff > 1e-10:
            print('Invalid :/')
            print('b: ')
            print(b)
            print('res: ')
            print(res)
            exit()

    print('Test left cancelation passed!')


def mobius_addition_cancel_test():
    for i in range(0, 10000):
        a = np.random.uniform(low=-0.01, high=0.01, size=10)

        res = torch_mob_add(-a, a, c=random.random())
        diff = np.sum(np.abs(res.cpu().numpy()))
        if diff > 1e-10:
            print('Invalid :/')
            print('res: ')
            print(res)
            exit()

    print('Test -a + a passed!')


def mobius_addition_2a_test():
    for i in range(0, 10000):
        a = np.random.uniform(low=-0.01, high=0.01, size=10)

        res1 = torch_mob_add(a, a, c=1.0)
        res2 = 2.0 / (1.0 + np.dot(a, a)) * a
        diff = np.sum(np.abs(res1.cpu().numpy() - res2))
        if diff > 1e-10:
            print('Invalid :/')
            print('res1: ')
            print(res1)
            print('res2: ')
            print(res2)
            exit()

    print('Test a+a passed!')


def mobius_addition_poinc_dist_test():
    for i in range(0, 10000):
        a = np.random.uniform(low=0.0, high=0.01, size=10)
        b = np.random.uniform(low=0.0, high=0.01, size=10)

        res1 = torch_poinc_dist_sq(torch.from_numpy(a).float(), torch.from_numpy(b).float(), c=1.0)
        res2 = 2 * np.arctanh(np.linalg.norm(torch_mob_add(-torch.from_numpy(a).float(), torch.from_numpy(b).float(), c=1.0).cpu().numpy()))
        diff = np.sum(np.abs(res1.cpu().numpy() - res2**2))
        if diff > 1e-10:
            print('Test 4 FAILED at trial %d :/' % i)
            print('res1: ')
            print(res1)
            print('res2: ')
            print(res2)
            print('2xres2: ')
            print(2 * res2)
            print('2xres2 - res1')
            print(2 * res2 - res1)
            return

    print('Test poinc dist - mobius passed!')


def mobius_addition_zero_b_test():
    for i in range(0, 10000):
        a = np.zeros(10)
        b = np.random.uniform(low=-0.01, high=0.01, size=10)

        res = torch_mob_add(torch.from_numpy(a).float(), torch.from_numpy(b).float(), c=1.0)
        diff = np.sum(np.abs(res.cpu().numpy() - b))
        if diff > 1e-10:
            print('Test 5 FAILED at trial %d :/' % i)
            print('res: ')
            print(res)
            print('b: ')
            print(b)
            exit()

    print('Test 0 + b passed!')


def mobius_addition_negative_test():
    for i in range(0, 10000):
        a = np.random.uniform(low=-0.01, high=0.01, size=10)
        b = np.random.uniform(low=-0.01, high=0.01, size=10)

        c = random.random()
        res1 = torch_mob_add(-torch.from_numpy(a).float(), -torch.from_numpy(b).float(), c)
        res2 = -torch_mob_add(torch.from_numpy(a).float(), torch.from_numpy(b).float(), c)
        diff = np.sum(np.abs(res1.cpu().numpy() - res2.cpu().numpy()))

        if diff > 1e-10:
            print('Test 6 FAILED at trial %d :/' % i)
            print('res1: ')
            print(res1)
            print('res2: ')
            print(res2)
            exit()

    print('Test a+b = -a + -b passed!')


def mobius_addition_infinity_test():
    for i in range(0, 10000):
        a = np.random.uniform(low=-0.01, high=0.01, size=10)
        b = np.random.uniform(low=-0.01, high=0.01, size=10)

        a = a / LA.norm(a)

        res = torch_mob_add(torch.from_numpy(a).float(), torch.from_numpy(b).float(), c=1.0)
        diff = LA.norm(a - res.cpu().numpy())

        if diff > 1e-10:
            print('Test 7 FAILED at trial %d :/' % i)
            print('res: ')
            print(res)
            print('diff: ')
            print(diff)
            exit()

        res = torch_mob_add(torch.from_numpy(b).float(), torch.from_numpy(a).float(), c=1.0)
        diff = np.abs(1 - LA.norm(res.cpu().numpy()))

        if diff > 1e-10:
            print('Test 7 FAILED at trial %d :/' % i)
            print('res: ')
            print(res)
            print('diff: ')
            print(diff)
            exit()


    print('Test mob add at infinity passed!')

