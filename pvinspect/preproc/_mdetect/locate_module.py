import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage.filters import gaussian_filter1d
import math
from .config import GAUSSIAN_RELATIVE_SIGMA, OUTER_CORNER_THRESH_FACTOR
from scipy import optimize
from pvinspect.common import transform
from .summary import Summary

summary = Summary('locate_module')

def _find_stops(img, dim):
    img = img.T if dim == 0 else img

    # calculate downsampling
    size = 0.5*np.sum(img.shape)
    #sigma = GAUSSIAN_RELATIVE_SIGMA*size
    #s = np.floor(2/3 * np.pi * sigma)
    #sigma_s = (3*s) / (2*np.pi)
    #if sigma**2 - sigma_s**2 < 0.5**2:
    #    # convince Nyquist
    #    s -= 1
    #    sigma_s = (3*s) / (2*np.pi)
    #sigma_filter = np.sqrt(sigma**2 - sigma_s**2)

    #print(sigma, s, sigma_filter)

    # extract profile of cumsum along di
    profile = np.sum(img, 1)
    profile_smooth = gaussian_filter1d(profile, GAUSSIAN_RELATIVE_SIGMA*size)
    profile_smooth = profile_smooth-np.min(profile_smooth)
    profile_smooth = profile_smooth/(np.max(profile_smooth)+1e-5)

    # calculate gradient of that
    grad_smooth = np.gradient(profile_smooth)

    # find maxima and minima and lower threshold if none found
    for i in range(1,10):
        thresh = (OUTER_CORNER_THRESH_FACTOR/i)*np.std(grad_smooth)
        maxima = np.argwhere(grad_smooth > thresh).flatten().tolist()
        minima = np.argwhere(grad_smooth < -thresh).flatten().tolist()
        #print(thresh)
        #print(maxima)
        #print(minima)

        if len(maxima) > 0 and len(minima) > 0:
            break

    #plt.plot(np.arange(profile.shape[0]), grad_sharp)
    #plt.show()
    #plt.plot(np.arange(profile.shape[0]), grad_smooth)
    #plt.show()
    if len(maxima) == 0 or len(minima) == 0:
        return None

    # non maximum/minimum supression
    max_last = maxima[0] < minima[0]
    maxi = 1 if max_last else 0
    mini = 0 if max_last else 1
    while True:
        if max_last:
            # expect minimum next
            if maxima[maxi] < minima[mini]:
                # not true: only keep highest maximum
                if grad_smooth[maxima[maxi-1]] < grad_smooth[maxima[maxi]]:
                    del maxima[maxi-1]
                else:
                    del maxima[maxi]
            else:
                # ok: go on
                max_last = False
                mini += 1
                #maxi += 1
        else:
            # expect maximum next
            if maxima[maxi] > minima[mini]:
                # not true: only keep smallest minimum
                if grad_smooth[minima[mini-1]] > grad_smooth[minima[mini]]:
                    del minima[mini-1]
                else:
                    del minima[mini]
            else:
                # ok: go on
                max_last = True
                #mini += 1
                maxi += 1
        if mini < len(minima) and maxi >= len(maxima) and len(minima)-mini > 1:
            # more than 1 minimum remaining
            idx = np.argmin(np.array(grad_smooth)[minima[mini:]])
            minima = minima[:mini] + [minima[mini+idx]]
        if maxi < len(maxima) and mini >= len(minima) and len(maxima)-maxi > 1:
            # more than 1 maximum remaining
            idx = np.argmax(np.array(grad_smooth)[maxima[maxi:]])
            maxima = maxima[:maxi] + [maxima[maxi+idx]]

        if mini >= len(minima) or maxi >= len(maxima):
            break

    # max -> min pairs
    if minima[0] < maxima[0]:
        minima = minima[1:]
    if len(maxima) > len(minima):
        maxima = maxima[:-1]

    if len(maxima) == 0 or len(minima) == 0:
        return None

    # length of pairs
    l = np.array(minima)-np.array(maxima)

    # take largest
    maxi = np.argmax(l)
    extremals = [maxima[maxi], minima[maxi]]

    #extremals = [np.argmax(grad_smooth), np.argmin(grad_smooth)]

    thresh = np.std(grad_smooth)*OUTER_CORNER_THRESH_FACTOR
    res = []
    res.append(extremals[0] - np.argmax((grad_smooth <= thresh)[extremals[0]::-1]))
    res.append(extremals[0] + np.argmax((grad_smooth <= thresh)[extremals[0]::+1]))
    res.append(extremals[1] - np.argmax((grad_smooth >= -thresh)[extremals[1]::-1]))
    res.append(extremals[1] + np.argmax((grad_smooth >= -thresh)[extremals[1]::+1]))

    # store summary
    dimstr = 'x' if dim == 0 else 'y'
    summary.put('profile_{}'.format(dimstr), profile)
    summary.put('profile_grad_smooth_{}'.format(dimstr), grad_smooth)
    summary.put('stops_{}'.format(dimstr), res)

    return res


def _assign_stops(x_stops, y_stops, img):

    # find outer and inner bounding box
    outer_anchor = (min(x_stops[0], x_stops[1]), min(y_stops[0], y_stops[1]))
    inner_anchor = (max(x_stops[0], x_stops[1]), max(y_stops[0], y_stops[1]))
    outer_size = (max(x_stops[2], x_stops[3])-outer_anchor[0]+1, max(y_stops[2], y_stops[3])-outer_anchor[1]+1)
    inner_size = (min(x_stops[2], x_stops[3])-inner_anchor[0]+1, min(y_stops[2], y_stops[3])-inner_anchor[1]+1)
    #fig, ax = plt.subplots(1)
    #ax.imshow(img, cmap='gray')
    #rect_outer = patches.Rectangle(outer_anchor, *outer_size, linewidth=1, edgecolor='r', facecolor='none')
    #rect_inner = patches.Rectangle(inner_anchor, *inner_size, linewidth=1, edgecolor='r', facecolor='none')
    #ax.add_patch(rect_outer)
    #ax.add_patch(rect_inner)
    #plt.show()

    # find boxes between outer and inner
    upper_margin = (max(x_stops[0], x_stops[1]), min(y_stops[0], y_stops[1]), min(x_stops[2], x_stops[3]), max(y_stops[0], y_stops[1]))
    left_margin = (min(x_stops[0], x_stops[1]), max(y_stops[0], y_stops[1]), max(x_stops[0], x_stops[1]), min(y_stops[2], y_stops[3]))
    bottom_margin = (max(x_stops[0], x_stops[1]), min(y_stops[2], y_stops[3]), min(x_stops[2], x_stops[3]), max(y_stops[2], y_stops[3]))
    right_margin = (min(x_stops[2], x_stops[3]), max(y_stops[0], y_stops[1]), max(x_stops[2], x_stops[3]), min(y_stops[2], y_stops[3]))

    # divide them into left/right or top/bottom part
    upper_margin_l = (upper_margin[1], upper_margin[0], upper_margin[3], upper_margin[0]+(upper_margin[2]-upper_margin[0]+1)//2)
    upper_margin_r = (upper_margin[1], upper_margin_l[3], upper_margin[3], upper_margin[2])
    left_margin_t = (left_margin[1], left_margin[0], left_margin[1]+(left_margin[3]-left_margin[1]+1)//2, left_margin[2])
    left_margin_b = (left_margin_t[2], left_margin_t[1], left_margin[3], left_margin[2])
    bottom_margin_l = (bottom_margin[1], bottom_margin[0], bottom_margin[3], bottom_margin[0]+(bottom_margin[2]-bottom_margin[0]+1)//2)
    bottom_margin_r = (bottom_margin[1], bottom_margin_l[3], bottom_margin[3], bottom_margin[2])
    right_margin_t = (right_margin[1], right_margin[0], right_margin[1]+(right_margin[3]-right_margin[1]+1)//2, right_margin[2])
    right_margin_b = (right_margin_t[2], right_margin[0], right_margin[3], right_margin[2])

    summary.put('box_upper_l', upper_margin_l)
    summary.put('box_upper_r', upper_margin_r)
    summary.put('box_left_t', left_margin_t)
    summary.put('box_left_b', left_margin_b)
    summary.put('box_bottom_l', bottom_margin_l)
    summary.put('box_bottom_r', bottom_margin_r)
    summary.put('box_right_t', right_margin_t)
    summary.put('box_right_b', right_margin_b)

    def sum_helper(img, p0, p1):
        s0 = int(np.ceil((p1[0]-p0[0]) / 10))       # sample approx. 10 lines per dim
        s1 = int(np.ceil((p1[1]-p0[1]) / 10))
        s0 = s0 if s0 > 0 else 1
        s1 = s1 if s1 > 0 else 1
        return np.sum(img[p0[0]:p1[0]+1:s0, p0[1]:p1[1]+1:s1])

    # sum over parts
    sum_upper_margin_l  = sum_helper(img, upper_margin_l[:2], upper_margin_l[2:])
    sum_upper_margin_r  = sum_helper(img, upper_margin_r[:2], upper_margin_r[2:])
    sum_left_margin_t   = sum_helper(img, left_margin_t[:2], left_margin_t[2:])
    sum_left_margin_b   = sum_helper(img, left_margin_b[:2], left_margin_b[2:]) 
    sum_bottom_margin_l = sum_helper(img, bottom_margin_l[:2], bottom_margin_l[2:])
    sum_bottom_margin_r = sum_helper(img, bottom_margin_r[:2], bottom_margin_r[2:]) 
    sum_right_margin_t  = sum_helper(img, right_margin_t[:2], right_margin_t[2:])
    sum_right_margin_b  = sum_helper(img, right_margin_b[:2], right_margin_b[2:])

    # assign stops
    if sum_upper_margin_l > sum_upper_margin_r:
        Ay = y_stops[0]
        By = y_stops[1]
    else:
        Ay = y_stops[1]
        By = y_stops[0]

    if sum_bottom_margin_l > sum_bottom_margin_r:
        Cy = y_stops[2]
        Dy = y_stops[3]
    else:
        Cy = y_stops[3]
        Dy = y_stops[2]

    if sum_left_margin_t > sum_left_margin_b:
        Ax = x_stops[0]
        Dx = x_stops[1]
    else:
        Ax = x_stops[1]
        Dx = x_stops[0]

    if sum_right_margin_t > sum_right_margin_b:
        Bx = x_stops[3]
        Cx = x_stops[2]
    else:
        Bx = x_stops[2]
        Cx = x_stops[3]

    return np.array([(Ax, Ay), (Bx, By), (Cx, Cy), (Dx, Dy)])


def locate_module(img):
    x_stops = _find_stops(img, 0)
    y_stops = _find_stops(img, 1)

    if x_stops is None or y_stops is None:
        return None
    else:
        return _assign_stops(x_stops, y_stops, img)


def module_boundingbox_model(coords, n_cols, m_rows):
    mean_x = 1/2 * (coords[1,0]-coords[0,0] + coords[2,0]-coords[3,0])
    mean_y = 1/2 * (coords[2,1]-coords[1,1] + coords[3,1]-coords[0,1])
    oriented_horizontal = mean_x > mean_y

    if oriented_horizontal:
        src = np.array([[0,0], [n_cols,0], [n_cols,m_rows], [0,m_rows]])
    else:
        src = np.array([[0,m_rows], [0,0], [n_cols,0], [n_cols,m_rows]])

    return src


#def refine_location(coords, iimg, iimg2):
#    a_iimg = np.prod(iimg.shape)
#    avg_size = np.sum(iimg.shape)/2
#    s_iimg = iimg[-1,-1]
#    s2_iimg = iimg2[-1,-1]
#
#    def obj(x):
#        x = x.reshape(-1,2)
#        path = integral_tools.create_path(x, 4)
#        s_inner, a_inner = integral_tools.shape_sum(path, iimg, True, 'linear')
#        s2_inner = integral_tools.shape_sum(path, iimg2, False, 'linear')
#        s_outer = s_iimg-s_inner
#        s2_outer = s2_iimg-s2_inner
#        a_outer = a_iimg-a_inner
#        m_inner = s_inner/a_inner
#        m_outer = s_outer/a_outer
#        m2_inner = s2_inner/a_inner
#        m2_outer = s2_outer/a_outer
#        std_inner = np.sqrt(m2_inner-m_inner**2)
#        std_outer = np.sqrt(m2_outer-m_outer**2)
#        res = ( (std_inner+std_outer) / (m_inner-m_outer) )**2
#        return res
#
#    x_max = iimg.shape[1]-1
#    y_max = iimg.shape[0]-1
#    bounds = [(0,x_max), (0,y_max)]*4
#    init = coords.flatten()
#
#    res = optimize.minimize(obj, init, bounds = bounds, method = 'SLSQP', options = dict(eps = 0.0000000001*avg_size))
#    print(res)
#
#    return res.x.reshape(-1,2)

    
