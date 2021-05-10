# %%
import os
import numpy as np
import cv2
import mediapy as media
import itertools as it
from scipy.spatial import KDTree
pi2 = np.pi*2



def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def computeGradient(blk):
    dx = 0.5*(blk[:, 2, 1, 1] - blk[:, 0, 1, 1])
    dy = 0.5*(blk[:, 1, 2, 1] - blk[:, 1, 0, 1])
    dz = 0.5*(blk[:, 1, 1, 2] - blk[:, 1, 1, 0])
    grad = np.transpose([dx, dy, dz], (1, 0))
    return grad

def computeHessian(blk):
    pix = blk[:, 1, 1, 1]
    dxx = blk[:, 2, 1, 1] - 2 * pix + blk[:, 0, 1, 1]
    dyy = blk[:, 1, 2, 1] - 2 * pix + blk[:, 1, 0, 1]
    dzz = blk[:, 1, 1, 2] - 2 * pix + blk[:, 1, 1, 0]
    dxy = 0.25 * (blk[:, 2, 2, 1] - blk[:, 2, 0, 1] - blk[:, 0, 2, 1] + blk[:, 0, 0, 1])
    dxz = 0.25 * (blk[:, 2, 1, 2] - blk[:, 2, 1, 0] - blk[:, 0, 1, 2] + blk[:, 0, 1, 0])
    dyz = 0.25 * (blk[:, 1, 2, 2] - blk[:, 1, 2, 0] - blk[:, 1, 0, 2] + blk[:, 1, 0, 0])
    hess = np.array([
        [dxx, dxy, dxz], 
        [dxy, dyy, dyz],
        [dxz, dyz, dzz]]).transpose(2, 0, 1)
    return hess

def SIFT(img, SIGMA=1.6, S=5, OCTAVE=4, show=None, contrast_threshold=.3):
    ''' 
    Return:
        ids:        (num of keypoints, 2): (i, j)
                    Discrete indices of keypoints in img
        desc:       (num of keypoints, 128)
                    128D-Discriptors
    '''
    # ## DoG in Scale-Space

    def normalize(img):
        min = img.min()
        ret = img.copy() - min
        ret /= ret.max()
        return ret
    if img.max() > 1.:
        img = img.astype(np.float32)/255
    else:
        img = img.copy()



    mult = 2**(1/(S-1))
    scale_imgs = []
    gs_imgs = []
    sigmas = []
    tar = cv2.GaussianBlur(img, (3, 3), SIGMA*.5)
    for o in range(OCTAVE):
        sigma = SIGMA * (2**o)
        gs_img = []
        for s in range(S):
            sigmas.append(sigma)
            gs_img.append(cv2.GaussianBlur(tar, (3, 3), sigma))
            sigma *= mult
        dog = np.diff(gs_img, axis=0)
        scale_imgs.append([normalize(i) for i in dog])
        gs_imgs.append(gs_img)
        tar = cv2.resize(tar, (tar.shape[1]//2, tar.shape[0]//2))
    sigmas = np.array(sigmas).reshape(OCTAVE, S).astype(np.float32)


    # for i in range(OCTAVE):
    #     media.show_images(scale_imgs[i])

    # ## Key Localization (extrema)

    # %%
    def detectExterema(simgs, threshold=0.6):
        simgs = np.array(simgs)
        slwds = np.lib.stride_tricks.sliding_window_view(simgs, (3, 3, 3))
        ids = []
        ids_orig = []
        removed = 0
        for idx_img, slwd in enumerate(slwds):
            slwd_flat = slwd.reshape(*slwd.shape[:2], -1)
            # (1, 1, 1) == np.unravel_index(13, (3, 3, 3))
            # That is, the center has index 13 in a 3x3x3 block
            select = slwd_flat.argmax(2) == 13
            if np.sum(select) == 0:
                continue
            # print(slwd.shape)
            # print(maxs.shape)
            # (..., scale, h, w)
            blk = slwd[select]
            # print(blk.shape)
            
            # Gradient
            grad = computeGradient(blk)
            # print(grad.shape)
            
            # Hessian
            hess = computeHessian(blk)
            # print(hess.shape)

            # Fix exterema position
            idx_fixed = -np.matmul(np.linalg.inv(hess), grad.reshape(grad.shape[0], 3, 1)).squeeze()
            idx_fix_orig = idx_fixed.copy()

            sel_pos = idx_fixed>.5
            sel_neg = idx_fixed<-.5
            idx_fixed[sel_pos] = 1
            idx_fixed[sel_neg] = -1
            idx_fixed[~(sel_pos|sel_neg)] = 0
            idx_fixed = idx_fixed
            
            idx = np.argwhere(select)+1
            idx = np.concatenate([np.full((len(idx), 1), idx_img), idx], 1)
            # print(idx.shape)
            idx_fixed = (idx_fixed+idx).astype(int)
            # print(idx_fixed)
            
            # Remove low contrast
            val = simgs[idx_img:idx_img+3][tuple(zip(*idx_fixed))] + 0.5*(grad*idx_fix_orig).sum(1)
            sel_low = val > threshold

            # Remove edges
            H = hess[:, 1:, 1:]
            tr = np.trace(H, axis1=1, axis2=2)
            det = np.linalg.det(H)
            sel_edge = (tr**2 / det) < 12.1

            # print(f"Edge: {np.sum(~sel_edge)}, Low contrast: {np.sum(~sel_low)}, Total: {sel_low.shape}")
            select = (sel_edge&sel_low)

            # print(f"Remove {(~select).sum()} points")
            removed += (~select).sum()
            ids += idx_fixed[select].tolist()
            ids_orig += (idx_fix_orig+idx)[select].tolist()
        print(f"Removed: {removed}, Remained: {len(ids)}")

        return np.array(ids).astype(int), np.array(ids_orig)


    def keyLocalization():
        ids = []
        ids_orig = []
        for o in range(OCTAVE):
            oids, oids_orig = np.array(detectExterema(scale_imgs[o], contrast_threshold))
            if len(oids) == 0:
                print(f"Skip {o+1}")
                # ids.append([])
                # ids_orig.append([])
                continue
            # octave, sigma(idx), h, w
            oids = np.concatenate([np.full((oids.shape[0], 1), o), oids], 1)
            oids_orig = np.concatenate([np.full((oids_orig.shape[0], 1), o), oids_orig], 1)
            ids.append(oids)
            ids_orig.append(oids_orig)
            # print(f"Octave {o+1}: {len(oids)}")
        ids = np.concatenate(ids, axis=0).astype(int)
        ids_orig = np.concatenate(ids_orig, axis=0)
        return ids, ids_orig
    
    ids, ids_orig = keyLocalization()

    # %%
    def drawKeyLocal():
        for o in range(OCTAVE):
            p = show.copy()
            id = ids_orig[ids_orig[:, 0]==o]
            # print(id)
            if len(id) == 0:
                continue
            for o, s, i, j in id:
                # color = (255, 255//s ,255//s)
                color = (255, 0 ,0)
                ii, jj = int(i*2**o), int(j*2**o)
                cv2.drawMarker(p, (jj, ii), color, markerSize=5)
                cv2.circle(p, (jj, ii), int(2*2**o), color, 1)
            media.show_image(p)
            # media.show_image(cv2.rotate(p, cv2.ROTATE_90_COUNTERCLOCKWISE))
            # media.show_image(rotate_image(p, -30))
    if show: drawKeyLocal
        
    # Orientation Assignment

    half_width = 8
    cell_width = half_width*2
    k = np.array([[-1, 0, 1]])
    mags = []
    angs = []
    # dxs = []
    # dys = []
    for gs_imgs_octave in gs_imgs:
        mag = []
        ang = []
        # dxx = []
        # dyy = []
        for gs_img in gs_imgs_octave[:-1]:
            dx = (cv2.filter2D(gs_img, -1, k, ))
            dy = (cv2.filter2D(gs_img, -1, k.T))
            # dxx.append(dx)
            # dyy.append(dy)
            ang_ = cv2.copyMakeBorder(np.arctan2(dx, dy), half_width, half_width-1, half_width, half_width-1, cv2.BORDER_REFLECT)
            ang_[ang_<0] += pi2
            ang.append(ang_)
            mag.append(cv2.copyMakeBorder(np.sqrt(dx**2+dy**2), half_width, half_width-1, half_width, half_width-1, cv2.BORDER_REFLECT))
            # mag.append(np.sqrt(dx**2+dy**2))
            # ang.append(np.arctan2(dx, dy)+np.pi) # dx is vertical
        angs.append(ang)
        mags.append(mag)
        # dxs.append(dxx)
        # dys.append(dyy)


    # 
    cells_mag = []
    cells_ang = []
    for o, s, i, j in ids:
        cells_mag.append(mags[o][s][i:i+cell_width, j:j+cell_width])
        cells_ang.append(angs[o][s][i:i+cell_width, j:j+cell_width])

    def sampleRotatedCell(o, s, pi, pj, orient):
        offset = (cell_width*0.5-0.5)
        a = np.arange(0, cell_width)-offset
        y, x = np.meshgrid(a, a)
        theta = -orient
        sin  = np.sin(theta)
        cos  = np.cos(theta)
        xx = x*cos-y*sin
        yy = x*sin+y*cos

        mag = mags[o][s]

        yy = np.clip(np.round(pi+yy).astype(int)+half_width, 0, mag.shape[0]-1)
        xx = np.clip(np.round(pj+xx).astype(int)+half_width, 0, mag.shape[1]-1)
        # tuple(zip(*yy, xx))
        mag = mag[(yy, xx)]
        ang = np.remainder(angs[o][s][(yy, xx)]+theta, pi2)
        # print(ang.min(), ang.max())
        # print(mag.shape, ang.shape)
        return mag, ang

    # %%
    def sampleRelativeCell(ids, ids_orig, cells_mag, cells_ang, weight):
        new_cells_ang = []
        new_cells_mag = []
        new_ids = [] # mapped from ids
        orients = []
        # mags = []
        bin = np.histogram_bin_edges([], bins=36, range=(0, pi2))
        bin = (bin[:-1]+bin[1:])*.5
        for i, (o, s, pnti, pntj) in enumerate(ids):
            hist = np.histogram(cells_ang[i], bins=36, range=(0, pi2), weights=(cells_mag[i]+1e-5)*weight[o][s])[0]
            max_id = np.argmax(hist)
            near_peak = (hist > 0.8*hist[max_id])
            num_near_peak = near_peak.sum()
            if num_near_peak > 2:
                # abort
                continue
            else:
                for peak_id in np.argwhere(near_peak):
                    orient = bin[peak_id[0]]
                    mag, ang = sampleRotatedCell(o, s, pnti, pntj, orient)
                    new_cells_ang.append(ang)
                    new_cells_mag.append(mag)
                    new_ids.append(i)
                    orients.append(orient)

        new_cells_mag = np.array(new_cells_mag)
        new_cells_ang = np.array(new_cells_ang)
        new_ids_orig = ids_orig[new_ids]
        new_ids = ids[new_ids]
        return new_cells_mag, new_cells_ang, new_ids, new_ids_orig, orients
    
    weight = [[cv2.getGaussianKernel(cell_width, 1.*s) for s in ss] for ss in sigmas]
    weight = [[w@w.T for w in ww] for ww in weight]
    cells_mag, cells_ang, ids, ids_orig, orients = sampleRelativeCell(ids, ids_orig, cells_mag, cells_ang, weight)

    # %%
    def draw_arrow():
        show_ = mags[0][0][8:-8, 8:-8].copy()
        # show_ = show.copy()
        for idx, (o, s, i, j) in enumerate(ids_orig):
            j = int(j*2**o)
            i = int(i*2**o)
            # if o > 1:
            #     continue
            p1 = (j, i)
            ang = -orients[idx]
            mag = 10 *2**o
            c = mag*np.cos(ang)
            s = mag*np.sin(ang)
            p2 = (int(j - s), int(i + c))
            cv2.arrowedLine(show_, p1, p2, (.3, 0, 0), thickness=1)
        media.show_image(show_)
        media.show_image(rotate_image(show_, -45))
    if show: draw_arrow()


    # ## Local Image Descriptor
    # %%
    weight = [[cv2.getGaussianKernel(16, s) for s in ss] for ss in sigmas]
    weight = [[w@w.T for w in ww] for ww in weight]
    weighted_cells_mag = [cells_mag[i]*weight[o][s] for i, (o, s, _, _) in enumerate(ids)]
    # for i, (o, s, _, _) in enumerate(ids):
    #     weighted_cells_mag.append(cells_mag[i]*weight[o][s])
    weighted_cells_mag = np.array(weighted_cells_mag)

    # %%
    # print(weighted_cells_mag.reshape(-1, 4, 4, 4, 4).swapaxes(2, 3).reshape(-1, 16)[-1])
    desc = [np.histogram(ang, bins=8, range=(0, pi2), density=True, weights=mag+1e-6)[0]
        for ang, mag in zip(*[
            cells_ang.reshape(-1, 4, 4, 4, 4).swapaxes(2, 3).reshape(-1, 16),
            weighted_cells_mag.reshape(-1, 4, 4, 4, 4).swapaxes(2, 3).reshape(-1, 16)])]
    desc = np.array(desc)
    desc[desc>.2] = .2
    desc = (desc/np.linalg.norm(desc, ord=2, axis=1, keepdims=True)).reshape(-1, 128)

    # Calculate the quantized indices
    ids = (ids_orig[:, 2:]*(2**ids_orig[:, [0]]))
    return np.array(ids), desc

def queryPoints(pdesc, pids, min_matches=-1, ret_match_img=None):
    ''' 
    Input:
        pdesc:          descriptor pair (2, len(desc))
        pids:           points coordinate pair (2, len(desc), 2)
        ret_match_img:  (opt) img pair (2, h, w) for returning img from draw_matches

    Output:
        good_trainIdx:  cooresponded points idx, can be used as pids[0][good_trainIdx]
        good_queryIdx:  cooresponded points idx, can be used as pids[1][good_queryIdx]
        mimg:           (opt) returned img from draw_matches
    '''
    tree = KDTree(pdesc[0])
    dist, trainIdx = tree.query(pdesc[1], workers=8, k=2)
    threshold = 0.75
    sel = dist[:, 0] < threshold*dist[:, 1]
    while np.count_nonzero(sel) < min_matches and threshold < 0.95:
        print(f"Insufficient Matches: {np.count_nonzero(sel)}, threshold={threshold}")
        threshold += 0.05
        sel = dist[:, 0] < threshold*dist[:, 1]
    print(f"Good Matches: {np.count_nonzero(sel)}, threshold={threshold}")
    good_trainIdx = trainIdx[:, 0][sel]
    good_queryIdx = np.where(sel)[0]
    if ret_match_img:
        mimg = draw_matches(ret_match_img[0], pids[0][:, ::-1], ret_match_img[1], pids[1][:, ::-1],           good_trainIdx, good_queryIdx)
        return good_trainIdx, good_queryIdx, mimg
    return good_trainIdx, good_queryIdx

def draw_matches(img1, kp1, img2, kp2, trainIdx, queryIdx):
    ''' Draw matched points like cv2.draw_matches '''
    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1])

    new_img = np.zeros(new_shape, type(img1.flat[0]))  
    new_img[0:img1.shape[0],0:img1.shape[1]] = img1
    new_img[0:img2.shape[0],img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2
    
    r = 15
    thickness = 2
    for tid, qid in zip(*[trainIdx, queryIdx]):
        c = np.random.rand(3) if len(img1.shape) == 3 else np.random.rand(1)
        c = c.tolist()
        end1 = tuple(np.round(kp1[tid]).astype(int))
        end2 = tuple(np.round(kp2[qid]).astype(int) + np.array([img1.shape[1], 0]))
        cv2.line(new_img, end1, end2, c, thickness)
        cv2.circle(new_img, end1, r, c, thickness)
        cv2.circle(new_img, end2, r, c, thickness)
    return new_img