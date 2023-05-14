from typing import Optional
from typing import Tuple
from itertools import product

import numpy as np


def create_vw(
    y_train: np.ndarray, v_ref: np.ndarray, w_ref: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    create v and w from y_train, v_ref and w_ref

    Paremeters
    ----------
    y_train : numpy.array
        ovserved output data
    v_ref : numpy.array
        reference point at left lower
    w_ref : numpy.array
        reference point at right upper

    Returns
    -------
    v : numpy.array
        coordinates of each cell's left lower
    w : numpy.array
        coordinates of each cell's right upper
    """
    y_train_unique = np.unique(y_train, axis=0)
    pareto_front = find_pareto_only_y(y_train_unique)
    v_base = np.vstack((v_ref, pareto_front))
    v_base = v_base[v_base[:, -1].argsort(), :]
    # とりあえずグリッドを作成
    v_all = all_grid_cells(v_base)
    # パレートフロンティアに支配されていないvだけ残す
    # dominated_v, dominated_w = create_cells(pareto_front, w_ref)
    delete = not_dominated_cell_detect(v_all, pareto_front)
    v = np.delete(v_all, delete, axis=0)
    w_base = np.vstack((pareto_front, w_ref))
    w_base = w_base[w_base[:, -1].argsort(), :]
    w_all = all_grid_cells(w_base)
    # wも同様
    w = np.delete(w_all, delete, axis=0)

    return v, w


def find_pareto_only_y(y: np.ndarray) -> np.ndarray:
    """
    obtain only pareto frontier in y

    Parameters
    ----------
    y : numpy.array
        output data

    Returns
    -------
    pareto_front : numpy.array
        pareto frontier in y
    """
    y_copy = np.copy(y)
    pareto_front = np.zeros((0, y.shape[1]))
    i = 0

    while i < y_copy.shape[0]:
        y_outi = np.delete(y_copy, i, axis=0)
        # paretoだったら全部false
        flag = np.all(y_outi <= y_copy[i, :], axis=1)
        if not np.any(flag):
            pareto_front = np.append(pareto_front, [y_copy[i, :]], axis=0)
            i += 1
        else:
            y_copy = np.delete(y_copy, i, axis=0)
    return pareto_front


def calc_hypervolume(y: np.ndarray, w_ref: np.ndarray) -> float:
    """
    calculate pareto hypervolume

    Parameters
    ----------
    y : numpy.array
        output data
    w_ref : numpy.array
        reference point for calculating hypervolume

    Returns
    -------
    hypervolume : float
        pareto hypervolume
    """
    hypervolume = 0.0e0
    pareto_front = find_pareto_only_y(y)
    v, w = create_cells(pareto_front, w_ref)

    if v.ndim == 1:
        hypervolume = np.prod(w - v)
    else:
        hypervolume = np.sum(np.prod(w - v, axis=1))
    return hypervolume


def create_cells(
    pf: np.ndarray, ref: np.ndarray, ref_inv: Optional[np.ndarray] = None
):
    """
    N個のパレートフロンティアから支配された領域の排他的なセルの配列を作る. (最小化)

    Parameters
    ----------
    pareto frontier : numpy array
        pareto frontiers (N \times L)
    reference point : numpy array
        point that bound the objective upper space (L)
    reference point : numpy array
        point that bound the objective lower space (L) (for convinience of calculation)

    Retruns
    --------
    lower : numpy array
        lower position of M cells in region truncated by pareto frontier (M \times L)
    upper : numpy array
        upper position of M cells in region truncated by pareto frontier (M \times L)
    """
    N, L = np.shape(pf)

    if ref_inv is None:
        ref_inv = np.min(pf, axis=0)

    if N == 1:
        # 1つの場合そのまま返してよし
        return np.atleast_2d(pf), np.atleast_2d(ref)
    else:
        # refと作る超体積が最も大きいものをpivotとする
        hv = np.prod(pf - ref, axis=1)
        pivot_index = np.argmax(hv)
        pivot = pf[pivot_index]
        # print('pivot :', pivot)

        # pivotはそのままcellになる
        lower = np.atleast_2d(pivot)
        upper = np.atleast_2d(ref)

        # 2^Lの全組み合わせに対して再帰を回す
        for i in product(range(2), repeat=L):
            # 全て1のところにはパレートフロンティアはもう無い
            # 全て0のところはシンプルなセルになるので上で既に追加済
            iter_index = np.array(list(i)) == 0
            if (np.sum(iter_index) == 0) or (np.sum(iter_index) == L):
                continue

            # 新しい基準点(pivot座標からiの1が立っているところだけref座標に変換)
            new_ref = pivot.copy()
            new_ref[iter_index] = ref[iter_index]

            # 新しいlower側の基準点(計算の都合上) (下側基準点座標からiの1が立っているところだけpivot座標に変換)
            new_ref_inv = ref_inv.copy()
            new_ref_inv[iter_index] = pivot[iter_index]

            # new_refより全次元で大きいPareto解は残しておく必要あり
            new_pf = pf[(pf < new_ref).all(axis=1), :]
            # new_ref_invに支配されていない点はnew_refとnew_ref_invの作る超直方体に射影する
            new_pf[new_pf < new_ref_inv] = np.matlib.repmat(
                new_ref_inv, new_pf.shape[0], 1
            )[new_pf < new_ref_inv]

            # 再帰
            if np.size(new_pf) > 0:
                child_lower, child_upper = create_cells(
                    new_pf, new_ref, new_ref_inv
                )

                lower = np.r_[lower, np.atleast_2d(child_lower)]
                upper = np.r_[upper, np.atleast_2d(child_upper)]

    return lower, upper


def all_grid_cells(points: np.ndarray) -> np.ndarray:
    points_list = []
    for i in range(points.shape[1]):
        points_list.append(np.unique(points[:, i]))
    points_mesh = np.meshgrid(*points_list)
    all_grid = points_mesh[0].ravel()[:, None]
    for i in range(1, points.shape[1]):
        all_grid = np.hstack(
            (np.atleast_2d(all_grid), points_mesh[i].ravel()[:, None])
        )
    return all_grid


def not_dominated_cell_detect(
    vec: np.ndarray, pareto_front: np.ndarray
) -> Tuple[np.ndarray, ...]:
    vec_tile = np.tile(vec[:, np.newaxis, :], (1, pareto_front.shape[0], 1))
    pareto_front_tile = np.tile(
        pareto_front[np.newaxis, :, :], (vec.shape[0], 1, 1)
    )
    true_tile = np.all(np.any(vec_tile < pareto_front_tile, axis=2), axis=1)
    return np.where(true_tile == False)
