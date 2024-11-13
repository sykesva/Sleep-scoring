from neo import PlexonIO, NeuroExplorerIO
import h5py
import os
import numpy as np
from time import time
from sklearn.metrics import calinski_harabasz_score as CHS, silhouette_score as SH, davies_bouldin_score as DBS
from scipy.stats import zscore, percentileofscore, kurtosis
import scipy.signal as ss
import warnings
from tqdm import tqdm
from sklearn.mixture import GaussianMixture as GM
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
from sklearn.covariance import EllipticEnvelope as EE
import pandas as pd
warnings.simplefilter('ignore')


def open_plx(fname):
    file_hdf = fname[:-3] + 'hdf'
    if os.path.exists(file_hdf):
        with h5py.File(file_hdf, 'r') as f:
            signal = np.asarray(f['eeg'], np.float32)
            sf = 1000
            n_ch = signal.shape[1]
    else:
        t1 = time()
        if fname.endswith('plx'):
            segm = PlexonIO(fname).read_segment() 
        elif fname.endswith('nex'):
            segm = NeuroExplorerIO(fname).read_segment()
        signal = [asig for asig in segm.analogsignals if len(asig) > 0]
        sf = int(signal[-1].sampling_rate)
        print('plx-file opened for {:.1f} mins'.format((time()-t1)/60))
        n_ch = len(signal)
        signal = np.asarray(np.multiply(np.concatenate(signal, axis=1), 1000), np.float32)
        with h5py.File(file_hdf, 'w') as f:
            f.create_dataset('eeg', data=signal)
            f.create_dataset('sf', data=(sf, ))
    
    hours = signal.shape[0] / sf / 3600
    print('{:.8f} hours of data loaded: {} channels, sampling rate = {}'.format(hours, n_ch, sf))
    return signal.T

def prepare_delta(data, dband, window_sec=5, sf=1000, sm_w=25):
    # eps = np.vstack([data[i*window_sec*sf:(i+1)*window_sec*sf].reshape((1, -1)) for i in range(data.size//(window_sec*sf))])
    # fs, spec = ss.welch(eps, fs=sf, axis=1, nperseg=1024, noverlap=300, nfft=2048)
    deltaw = extract_fft_amps(data, [dband], window_sec=window_sec, sf=sf)[0]
    # deltaw = np.sum(np.log10(spec[:, (fs>dband[0])&(fs<dband[1])]), axis=1)
    deltawsm = np.convolve(deltaw, np.ones(sm_w)/sm_w, mode='same')
    deltawsm[:(sm_w//2+1)] = deltaw[sm_w//2+1]
    deltawsm[-(sm_w//2+1):] = deltaw[-(sm_w//2+1)]
    return deltaw, deltawsm

def vis_delta_res(delta_sm, delta_thrs):
    plt.rcParams['figure.figsize'] = (25, 4)
    plt.plot(delta_sm)
    plt.xlim(0, len(delta_sm))
    part = 1 / len(delta_thrs)
    for i, thr in enumerate(delta_thrs):
        plt.axhline(thr, xmin=i*part, xmax=(i+1)*part)
    plt.ylabel('Amplitude')
    plt.xlabel('Epochs #')
    plt.show()

def art_thr(art_rms):
    n, b, p = plt.hist(art_rms, 200)
    plt.close()
    b = (b[1:]+b[:-1])/2
    center = b[np.argmax(n[:150])]
    cont = .01
    thr = b[-1]
    # print('start', center, thr)
    kurts, thrs = [], []
    while (center < thr) and (cont < .5):
        y = EE(contamination=cont).fit_predict(art_rms.reshape((-1, 1)))
        kurts.append(kurtosis(art_rms[y == 1]))
        try: thrs.append((np.min(art_rms[y == -1]) + np.max(art_rms[y == 1]))/2)
        except: thrs.append(np.min(art_rms)+(np.max(art_rms) - np.min(art_rms))*.8)
        thr = thrs[-1]
        cont += .01
    idx = np.argmin(np.abs(kurts))
    plt.rcParams['figure.figsize'] = (25, 4)
    plt.plot(art_rms)
    plt.xlim(0, len(art_rms))
    plt.xlabel('Epoch #')
    plt.ylabel('Amplitude')
    plt.axhline(thrs[idx])
    plt.show()
    return thrs[idx], art_rms < thrs[idx]

def gen_d_s_bands(delta_freqs=(0, 6), sigma_freqs=(9.5, 16)):
    dbands = [(start, start+step) for start in np.arange(delta_freqs[0], delta_freqs[1], .5) for step in np.arange(1, delta_freqs[1]-start+.5, .5) if (start + step) <= delta_freqs[1]]
    sbands = [(start, start+step) for start in np.arange(sigma_freqs[0], sigma_freqs[1], .5) for step in np.arange(1, sigma_freqs[1]-start+.5, .5) if (start + step) <= sigma_freqs[1]]
    return dbands, sbands

def dc_remove(data, window_sec=2, fs=0.2):
    i, res, baseline = 0, [], []
    window = round(window_sec * fs)
    while i < len(data):
        baseline.append(np.min(data[max(0, i-window): min(len(data)-1, i+window)]))
        res.append(data[i] - baseline[-1])
        i += 1
    return np.array(res).flatten()

def prepare_ratio(data, tband, dseries, window_sec=5, sf=1000, sm_w=5, dc_w=100):
    tseries = extract_fft_amps(data, [tband], window_sec=window_sec, sf=sf)[0]
    ratio = np.convolve(tseries/dseries, np.ones(sm_w)/sm_w, mode='same')
    ratio[ratio < 0] = 0
    ratio = dc_remove(ratio, dc_w)
    ratio[ratio < 0] = 0
    return ratio

def gen_t_bands(theta_freqs):
    return [(start, start+step) for start in np.arange(theta_freqs[0], theta_freqs[1], .5) for step in np.arange(1, theta_freqs[1]-start+.5, .5) if (start + step) <= theta_freqs[1]]

def ratios_search(tseries, denominator, sm_w=5, dc_w=100):
    return [zscore(dc_remove(np.convolve(t/denominator, np.ones(sm_w)/sm_w, mode='same'), 100)) for t in tseries] # 

def ratio_metric(ratios):
    return [np.percentile(r, 99) - np.median(r) for r in ratios]

def extract_fft_amps(data, freqs_set, window_sec=5, sf=1000):
    eps = np.vstack([data[i*window_sec*sf:(i+1)*window_sec*sf].reshape((1, -1)) for i in range(data.size//(window_sec*sf))])
    fs, spec = ss.welch(eps, fs=sf, axis=1, nperseg=1024, noverlap=300, nfft=2048)
    spec = np.log10(spec)
    res = []
    for freqs in freqs_set:
        res.append(np.sum(spec[:, (fs >= freqs[0]) & (fs <= freqs[1])], axis=1))
    return res

def cluster_qual(labels, X):
    X = zscore(X)
    return CHS(X, labels), SH(X, labels), DBS(X, labels)

def gm_delta(delta):
    # print(delta.shape, delta)
    gm_model = GM(n_components=2, covariance_type='full', reg_covar=1e-3).fit(delta)
    gm_preds = gm_model.predict(delta)
    if gm_model.means_[0, 0] > [gm_model.means_[1, 0]]:
        gm_preds = (gm_preds-.5)*(-1)+.5
    return gm_preds.flatten()

def cluster_metrics_d_s(series):
    metrics = np.zeros((len(series), 3))
    for i in tqdm(range(len(series))):
        X = series[i]
        labels = gm_delta(X.reshape((-1, 1))).reshape((-1, 1))
        res = cluster_qual(labels, X.reshape((-1, 1)))
        metrics[i] = res
    return metrics

def vis_d_s_metrics(metrics, dbands, sbands):
    color = cm.coolwarm(np.linspace(0, 1, 101))
    for i, name in enumerate(('Calinski-Harabasz score↑', 'Silhouette score↑', 'Davies-Bouldin score↓')):
        plt.rcParams['figure.figsize'] = (25, 4)
        plt.title(name, fontsize=15)
        for x, y in zip(np.arange(len(metrics)), metrics[:, i]):
            plt.bar(x, y, color=color[int(percentileofscore(metrics[:, i], y))])

        plt.ylabel('Metric value')
        plt.xlabel('Band, Hz')
        plt.xticks(np.arange(len(metrics)), [f'{band[0]}-{band[1]}' for band in dbands+sbands], rotation=90)
        plt.axvline(len(dbands), c='k')
        plt.show()
        
def vis_ratio_metric(tmetrics, tbands):
    color = cm.coolwarm(np.linspace(0, 1, 101))
    plt.title('median-99% range', fontsize=15)
    for x, y in zip(np.arange(len(tmetrics)), tmetrics):
        plt.bar(x, y, color=color[int(percentileofscore(tmetrics, y))])
    plt.ylabel('Metric value')
    plt.xlabel('Band, Hz')
    plt.xticks(np.arange(len(tmetrics)), [f'{band[0]}-{band[1]}' for band in tbands], rotation=90)
    plt.axvline(len(tbands), c='k')
    plt.show()

def gm_delta_scoring(delta_, cycles=2, mask=slice(None, None)):
    delta_ = delta_[mask]
    points_cycle = int(delta_.size/cycles)
    res = []
    for c in range(cycles):
        delta = delta_[c*points_cycle:(c+1)*points_cycle].reshape((-1, 1))
        gm_model = GM(n_components=2, covariance_type='full', reg_covar=1e-3).fit(delta)
        gm_preds = gm_model.predict(delta)
        shift = 0
        if gm_model.means_[0, 0] > [gm_model.means_[1, 0]]:
            shift = 1
            gm_preds = (gm_preds-.5)*(-1)+.5
        res.append((np.max(delta[gm_preds==0])+np.min(delta[gm_preds==1]))/2)
    return res

def no_singles(ser, val):
    res = [ser[0]]
    for i in range(1, len(ser)-1):
        if (ser[i] == val) and (ser[i-1] == ser[i+1]) and (ser[i-1] != val):
            res.append(ser[i-1])
        else:
            res.append(ser[i])
    res.append(ser[-1])
    return res

def no_rems_in_wake(hypno, n_back):
    res = [*hypno[:n_back]]
    i = n_back
    while i < len(hypno):
        if (hypno[i] == 2) and (np.sum(hypno[i-n_back:i]) == 0): 
            res.append(0)
            j = i + 1
            while hypno[j] == 2:
                res.append(0)
                j += 1
                if j == len(hypno): break
            i = j
        else: 
            res.append(hypno[i])
            i += 1
    return np.array(res)

def no_single_a_between_b_and_c(hypno, a, b, c):
    res = [hypno[0]]
    for i in range(1, len(hypno)-1):
        if (hypno[i] == a) and (hypno[i-1] == b) and (hypno[i+1] == c):
            res.append(b)
        else:
            res.append(hypno[i])
    res.append(hypno[-1])
    return np.array(res) 

def stage(norm_mask, delta, ratio, delta_thrs, rem_thr):
    hypno = np.zeros(len(norm_mask))
    n_cycles = len(delta_thrs)
    n_pts_cycle = int(len(norm_mask)/n_cycles)
    for c in range(n_cycles):
        cycle_slice = slice(c*n_pts_cycle, (c+1)*n_pts_cycle)
        art_mask = norm_mask[cycle_slice]
        part_hypno = (delta[cycle_slice][art_mask] > delta_thrs[c]).astype(int)
        mask2 = (ratio[cycle_slice][art_mask] > rem_thr).astype(int)
        part_hypno[(part_hypno == 0) & (mask2 == 1)] = 2
        hypno[cycle_slice][art_mask] = part_hypno
    hypno = no_singles(hypno, 0)
    hypno = no_singles(hypno, 1)
    hypno = no_rems_in_wake(hypno, 2)
    hypno = no_single_a_between_b_and_c(hypno, 0, 1, 2)
    hypno = no_single_a_between_b_and_c(hypno, 1, 2, 2)
    return hypno

def prepare_scores(fname):
    scores = pd.read_excel(fname, usecols='B', skiprows=1)
    scores = np.array(scores).flatten()
    mapping = {'аб': 0, 'бс': 2, 'мс': 1, 'сб': 0}
    scores = np.array(list(map(lambda el: mapping[el], scores)))
    return scores