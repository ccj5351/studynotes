import matplotlib.pyplot as plt
import numpy as np
import sys
import os


def write_to_file(txt_fn, files_list):
    with open(txt_fn, 'w') as fw:
        for l in files_list:
            fw.write(l + "\n")
    print("Done! Saved {} names to {}".format(len(files_list), txt_fn))

def merge_hist_bins(hist, bin_edges, \
    hist_value_thred = 1, # i.e., 1% if is_percentile True;
    is_percentile = False
    ):
    total = np.sum(hist)
    if is_percentile:
        hist_thred = int(total*hist_value_thred*0.01)
    else:
        hist_thred = int(hist_value_thred)
    print ("[***] hist_thred = ", hist_thred)
    assert len(hist) == len(bin_edges) - 1
    bin_dict = {}
    i_rightmost = 0 # right most bin edge of current i;
    for i in range(0, len(hist)):
        if i < i_rightmost:
            continue
        edge_left = bin_edges[i] # bin left edge;
        j = i
        tmp_hist_sum = 0 # sum of several bins we have considered
        while tmp_hist_sum < hist_thred and j < len(hist):
            tmp_hist_sum += hist[j]
            j += 1
            edge_right = bin_edges[j] # find next bin if not yet enough elements found;
        else: # save new bin, with its left and right edges as key, and the element number as value;
            bin_dict[(edge_left, edge_right)] = tmp_hist_sum
        
        # save right-most right-edge, so that we can 
        # skip many incomimg indices from next loops;
        i_rightmost = j 
    
    #---------------------------------    
    # print info of the new histogram;
    idx = 0
    new_hist = []
    new_bin_edges = [bin_edges[0]]
    for k , v in bin_dict.items():
        new_hist.append(v)
        new_bin_edges.append(k[1])
        print ("key {} : {}".format(k, v))
        idx += 1
    print ("[***] done, hist_thred = ", hist_thred)
    print ("[***] old bin # = {}, new bin # = {}".format(len(bin_edges), len(new_bin_edges)))
    # return new hist, bins, and thred we used;
    return np.array(new_hist), np.array(new_bin_edges), hist_thred

# plotting the histogram        
def show_hist(bin_edges, hist, fig_file = None):
    d_min = bin_edges[0]
    d_max = bin_edges[-1]
    d_num = len(bin_edges)
    fig, ax = plt.subplots()  #create figure and axes
    
    ## this is the right way we how to draw histogram
    plt.hist(x=bin_edges[:-1], bins=bin_edges, weights=hist) 
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('My Very Own Histogram')
    # Figure size in inches (default)
    plt.text(x=0.5, y=0.5, \
        s=r'$D_{min}=$'+"{}".format(d_min) + r', $D_{max}=$'+"{}".format(\
            d_max) + r', $D_{num}=$'+"{}".format(d_num), 
        transform=ax.transAxes)
    if fig_file:
        plt.savefig("./results/{}.png".format(fig_file))
        print ("saved ", "./results/{}.png".format(fig_file))
    plt.show()
    txt_fn = "./results/" + npz_file + ".csv"
    comment = "#right_bin_edge, hist_value"
    file_lists = [ "{},{}".format(i, j if j > 50 else 0.5) for (i,j) in zip(bin_edges[1:], hist)]
    file_lists = [comment] + file_lists
    write_to_file(txt_fn, file_lists)


""" 
How to run this file:
- cd this_project
- python3 main_debug_light.py

"""
if __name__ == "__main__":
    tmp_dir = '/mnt/Data/changjiang/code/manydepth-study/results'
    tmp_dir = './results'
    npz_file = 'scanet_hist_dmin0.10_dmax10.00_dnum256'
    npz_file = 'scanet_hist_dmin0.25_dmax10.00_dnum256'
    npz_file = 'scanet_hist_dmin0.25_dmax10.00_dnum96'
    npz_file = 'scanet_hist_dmin0.25_dmax10.00_dnum64'
    npz_file = 'scanet_hist_dmin0.25_dmax10.00_dnum128'
    npz_file = 'scanet_hist_dmin0.25_dmax10.00_dnum256_inv'
    npz_file = 'scanet_hist_dmin0.25_dmax10.00_dnum256'
    #npz_file = 'scanet_hist_dmin0.25_dmax10.00_dnum96_inv'
    #npz_file = 'scanet_hist_dmin0.25_dmax10.00_dnum64_inv'
    #npz_file = 'scanet_hist_dmin0.25_dmax10.00_dnum128_inv'
    inverse = False
    inv_str = ''
    if "_inv" in npz_file:
        inverse = True
        inv_str = '_inv'
    arr = np.load(os.path.join(tmp_dir, npz_file + ".npz"))
    hist = arr['hist']
    bin_edges = arr['bins_edges']
    if 0:
        show_hist(bin_edges, hist, is_save_fig = False)
    print ("hist: len = {}\n".format(len(hist), hist))
    print ("bin_edges = \n", bin_edges)
    if 0:
        txt_fn = "./results/" + npz_file + ".csv"
        comment = "#right_bin_edge, hist_value"
        file_lists = [ "{},{}".format(i, j if j > 50 else 0.5) for (i,j) in zip(bin_edges[1:], hist)]
        file_lists = [comment] + file_lists
        write_to_file(txt_fn, file_lists)
    if 1:
        #merge_hist_bins(hist, bin_edges, hist_value_thred=6000, is_percentile=False)
        new_hist, new_bin_edges, hist_thred = merge_hist_bins(hist, bin_edges, hist_value_thred=0.12, is_percentile=True)
        d_min, d_max = new_bin_edges[0], new_bin_edges[-1]
        d_num = len(new_bin_edges)

        fig_file = 'scanet_mergedbins_dmin{:.2f}_dmax{:.2f}_dnum{:d}_histThre{:d}{}'.format(d_min, d_max, d_num, hist_thred, inv_str)
        show_hist(bin_edges= new_bin_edges, hist= new_hist, fig_file=fig_file)
