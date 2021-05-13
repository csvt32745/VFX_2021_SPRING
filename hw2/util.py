import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def show(img):
    ''' Image showing function for ipynb '''
    i = img.astype('uint8')*255 if img.dtype == 'bool' else img
    # print(i.dtype, i.ndim)
    plt.figure(figsize=(15, 10))
    if img.ndim == 3:
        # color
        plt.imshow(img[..., ::-1], interpolation='none')
    else:
        plt.imshow(img, cmap='gray', interpolation='none')
        # plt.imshow(img, cmap='gray', vmin=0, vmax=255, interpolation='none')
    plt.show()

def anchor_points(img):
    ''' Select the Anchor points on 4 corners in clock-wise order '''
    def on_mouse(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            param[0].append([x, y])
    points_add= []

    WINDOW_NAME = "Select the Anchor points on 4 corners in clock-wise order"
    cv.namedWindow(WINDOW_NAME)
    cv.setMouseCallback(WINDOW_NAME, on_mouse, [points_add])
    while len(points_add) < 4:
        img_ = img.copy()
        for i, p in enumerate(points_add):
            # draw points on img_
            c = [0, 0, 0]
            c[i%3] = 255
            cv.circle(img_, tuple(p), 5, c, -1)
        cv.imshow(WINDOW_NAME, img_)

        key = cv.waitKey(20) % 0xFF
        if key == 27: break # exist when pressing ESC

    cv.destroyAllWindows()
    
    print('{} Points added'.format(len(points_add)))
    # np.save('a.npy', points_add)
    return points_add

def normalization(pnts):
    ''' Compute the transformation matrix for normalization '''
    mean = pnts.mean(axis=0)
    # avg(points' distance to center)
    dist = np.mean(np.sqrt(np.sum((pnts-mean)**2, axis=1)))
    scale = 2**0.5 / dist 
    # (p-c)*s
    # | s   -sx |
    # |   s -sy |
    # |      1 |
    T = np.eye(3)
    T[:2, -1] = -mean
    T[-1, -1] = 1/scale
    return scale * T

def get_homography(pnts1, pnts2, is_normalized=False):
    if is_normalized:
        T1 = normalization(pnts1)
        T2 = normalization(pnts2)
        pnts1 = homo_transform(T1, pnts1)
        pnts2 = homo_transform(T2, pnts2)
    
    template = np.zeros((2, 9))
    template[0, 5] = -1
    template[1, 2] = 1
    l = []
    for p1, p2 in zip(pnts1, pnts2):
        a = template.copy()
        a[0, 3:5] = -p1
        a[1, 0:2] = p1
        tmp = np.concatenate([p1, [1]])
        a[0, -3:] = p2[1]*tmp
        a[1, -3:] = -p2[0]*tmp
        l.append(a)
    A = np.vstack(l)
    v = np.linalg.svd(A, full_matrices=True)[2]
    # print(v.shape)
    # print(s)
    h = v[-1].reshape(3, 3)

    if is_normalized:
        h = np.linalg.inv(T2) @ h @ T1

    return h / h[-1, -1]

def homo_transform(h, x):
    """
    Input:  h: 3x3 homogeneous matrix, x: (N, 2) for N 2d-points
    Output: (N, 2) for N 2d-points
    """
    res = h@np.vstack([np.atleast_2d(x).T, np.ones(x.shape[0])])
    return (res[:2]/res[-1]).T

def warpping(h_inv, img, shape):
    '''
    Perform inverse warpping
    h_inv:  inversed homography
    img:    source image
    shape:  output image shape
    '''
    # u', v' => u, v
    uv = np.indices((shape[1], shape[0])).reshape(2, -1).T
    warpped_uv = homo_transform(h_inv, uv).T
    
    u0, v0 = np.floor(warpped_uv).astype(int)
    u1, v1 = u0+1, v0+1
    du, dv = np.atleast_2d(warpped_uv[0]-u0).T, np.atleast_2d(warpped_uv[1]-v0).T

    clip_u = lambda x: np.clip(x, 0, img.shape[1]-1)
    clip_v = lambda x: np.clip(x, 0, img.shape[0]-1)
    u0 = clip_u(u0)
    u1 = clip_u(u1)
    v0 = clip_v(v0)
    v1 = clip_v(v1)
    
    c_v0 = img[v0, u0]*(1-du) + img[v0, u1]*du
    c_v1 = img[v1, u0]*(1-du) + img[v1, u1]*du

    res = c_v0*(1-dv) + c_v1*dv
    res = res.astype('uint8').reshape((shape[1], shape[0], 3))
    return np.swapaxes(res, 0, 1)

def draw_matches(img1, kp1, img2, kp2, matches):
    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1])

    new_img = np.zeros(new_shape, type(img1.flat[0]))  
    new_img[0:img1.shape[0],0:img1.shape[1]] = img1
    new_img[0:img2.shape[0],img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2
    
    r = 15
    thickness = 5
    for m in matches:
        c = np.random.randint(0,256,3) if len(img1.shape) == 3 else np.random.randint(0,256)
        c = c.tolist()
        end1 = tuple(np.round(kp1[m.queryIdx].pt).astype(int))
        end2 = tuple(np.round(kp2[m.trainIdx].pt).astype(int) + np.array([img1.shape[1], 0]))
        cv.line(new_img, end1, end2, c, thickness)
        cv.circle(new_img, end1, r, c, thickness)
        cv.circle(new_img, end2, r, c, thickness)

    return new_img