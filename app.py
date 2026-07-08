import os, sys, math, io, json, warnings, time, urllib.request

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

import numpy as np
import joblib
import cv2
from flask import Flask, request, jsonify, render_template, send_file
from scipy.signal import convolve2d as _conv2d
from scipy.ndimage import gaussian_filter as _gauss_filter
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Validation Metrics (from notebook Cell 16) ─────────────────────────────
VALIDATION_METRICS = {
    "robust_svm": {
        "auc": 0.8605,
        "accuracy_05": 0.7720,
        "accuracy_tuned": 0.7749,
        "best_threshold": 0.490,
        "classification_report": {
            "clean": {"precision": 0.7700, "recall": 0.6995, "f1": 0.7330, "support": 1847},
            "stego": {"precision": 0.7734, "recall": 0.8307, "f1": 0.8010, "support": 2280},
        },
        "per_dataset": {
            "ALASKA2":     {"n": 817, "auc": 0.5437, "acc": 0.7148},
            "BOSSBASE":    {"n": 813, "auc": 0.5650, "acc": 0.5375},
            "IPHONE":      {"n": 572, "auc": 1.0000, "acc": 1.0000},
            "STEGANAYIS":  {"n": 595, "auc": 0.5915, "acc": 0.6286},
            "STEGO-PVD":   {"n": 393, "auc": 0.9749, "acc": 0.9364},
            "STEGOIMAGES": {"n": 389, "auc": 0.9694, "acc": 0.9152},
            "UCID":        {"n": 200, "auc": 0.0000, "acc": 0.9800},
        }
    },
    "urd": {
        "auc": 0.8738,
        "accuracy_05": 0.7744,
        "accuracy_tuned": 0.7759,
        "best_threshold": 0.570,
        "classification_report": {
            "clean": {"precision": 0.7508, "recall": 0.7423, "f1": 0.7465, "support": 1847},
            "stego": {"precision": 0.7931, "recall": 0.8004, "f1": 0.7968, "support": 2280},
        },
        "per_dataset": {
            "ALASKA2":     {"n": 817, "auc": 0.5304, "acc": 0.6671},
            "BOSSBASE":    {"n": 813, "auc": 0.5779, "acc": 0.5658},
            "IPHONE":      {"n": 572, "auc": 1.0000, "acc": 1.0000},
            "STEGANAYIS":  {"n": 595, "auc": 0.6701, "acc": 0.6218},
            "STEGO-PVD":   {"n": 393, "auc": 0.9814, "acc": 0.9415},
            "STEGOIMAGES": {"n": 389, "auc": 0.9785, "acc": 0.9306},
            "UCID":        {"n": 200, "auc": 0.0000, "acc": 0.9850},
        }
    }
}

# ── Feature Extraction (from notebook) ──────────────────────────────────────
def _gray(img):
    if img.ndim == 3:
        return np.dot(img[:,:,:3].astype(np.float64), [0.299, 0.587, 0.114])
    return img.astype(np.float64)

_SRM_FILTERS = {
    'f1h': np.array([[-1, 2, -1]], dtype=float) / 2.0,
    'f1v': np.array([[-1],[2],[-1]], dtype=float) / 2.0,
    'f1d': np.array([[-1,0,1],[0,0,0],[1,0,-1]], dtype=float) / 2.0,
    'f2h': np.array([[-1,2,-2,2,-1]], dtype=float) / 4.0,
    'f2v': np.array([[-1],[2],[-2],[2],[-1]], dtype=float) / 4.0,
    'f3h': np.array([[1,-3,3,-1]], dtype=float),
    'f3v': np.array([[1],[-3],[3],[-1]], dtype=float),
    'sq1': np.array([[-1,2,-1],[2,-4,2],[-1,2,-1]], dtype=float) / 4.0,
    'sq2': np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=float) / 4.0,
    'sq3': np.array([[-1,0,1],[0,0,0],[1,0,-1]], dtype=float) / 2.0,
}

def _srm_features(gray01, T=3):
    feats = []
    for k in _SRM_FILTERS.values():
        r = _conv2d(gray01, k, mode='same', boundary='symm')
        r_int = np.clip(np.round(r * 255).astype(int), -T, T)
        r_flat = r_int.flatten()
        for v_idx in range(2*T+1):
            v = v_idx - T
            mask = (r_flat[:-1] == v)
            co = r_flat[1:][mask]
            h = np.bincount(co + T, minlength=2*T+1).astype(float)
            h /= (h.sum() + 1e-9)
            feats.append(h)
    fv = np.concatenate(feats)
    n = np.linalg.norm(fv)
    return fv / n if n > 0 else fv

def _srm_rgb(arr, T=2):
    feats = []
    filters_sel = {k: _SRM_FILTERS[k] for k in ['f1h','f1v','sq2']}
    for ch in range(3):
        ch_arr = arr[:,:,ch].astype(float) / 255.0
        for k in filters_sel.values():
            r = _conv2d(ch_arr, k, mode='same', boundary='symm')
            r_int = np.clip(np.round(r*255).astype(int), -T, T)
            h = np.bincount(r_int.flatten() + T, minlength=2*T+1).astype(float)
            h /= h.sum() + 1e-9
            feats.append(h.std())
            feats.append(float(np.mean(np.abs(r_int))))
    return np.array(feats[:18])

def _lsb_entropy_feats(arr):
    feats = []
    for i in range(min(3, arr.shape[2])):
        for bit in range(4):
            bp = ((arr[:,:,i] >> bit) & 1).astype(np.float64)
            p1 = bp.mean(); p0 = 1-p1
            H = 0.0
            if p0>0: H -= p0*math.log2(p0)
            if p1>0: H -= p1*math.log2(p1)
            feats.append(H)
    while len(feats) < 12: feats.append(0.0)
    return np.array(feats[:12])

def _chi_square_feats(arr):
    feats = []
    for i in range(3):
        flat = arr[:,:,i].flatten().astype(int)
        hist = np.bincount(flat, minlength=256)
        chi2 = 0.0
        for k in range(128):
            a,b = hist[2*k], hist[2*k+1]; tot=a+b
            if tot==0: continue
            e=tot/2.0; chi2 += ((a-e)**2)/e + ((b-e)**2)/e
        feats.append(np.clip(chi2/1000.0, 0, 50))
    return np.array(feats)

def _moments_feats(arr):
    from scipy.stats import skew, kurtosis
    feats = []
    for i in range(3):
        flat = arr[:,:,i].flatten().astype(np.float64)
        feats.extend([flat.mean()/255, flat.std()/128,
                      float(np.clip(skew(flat),-5,5))/5,
                      float(np.clip(kurtosis(flat,fisher=True),-10,10))/10])
    return np.array(feats)

def _gradient_feats(gray):
    gy, gx = np.gradient(gray)
    gmag = np.sqrt(gx**2+gy**2)
    gnorm = np.clip((gmag/(gmag.max()+1e-9))*255,0,255).astype(int)
    hist = np.bincount(gnorm.flatten(),minlength=256).astype(float); hist/=hist.sum()
    ent = -np.sum(hist[hist>0]*np.log2(hist[hist>0]))
    lap_var = float(np.var(_gauss_filter(gray,1)-gray))
    return np.array([ent/8.0, min(lap_var/1000,1.0)])

def _fft_feats(gray):
    F=np.fft.fft2(gray); P=np.abs(np.fft.fftshift(F))**2
    h,w=P.shape; cy,cx=h//2,w//2
    Y,X=np.ogrid[:h,:w]; dist=np.sqrt((X-cx)**2+(Y-cy)**2).astype(int)
    max_r=min(cx,cy)
    radial=np.array([P[dist==r].mean() if (dist==r).sum()>0 else 0 for r in range(max_r)])
    freqs=np.arange(1,max_r); rp=radial[1:max_r]; valid=rp>0
    beta=-2.0
    if valid.sum()>10:
        c=np.polyfit(np.log(freqs[valid]+1e-9),np.log(rp[valid]+1e-9),1); beta=c[0]
    split=int(max_r*0.8); lo=radial[:split].sum(); hi=radial[split:].sum()
    hf_ratio=hi/(lo+hi+1e-9)
    return np.array([np.clip(beta/5,-1,1), hf_ratio])

def _rle_feats(gray_uint8):
    flat=gray_uint8.flatten()
    runs=1+int(np.sum(flat[1:]!=flat[:-1]))
    return np.array([runs/len(flat)])

def _color_corr_feats(arr):
    h=[np.bincount(arr[:,:,i].flatten(),minlength=256).astype(float) for i in range(3)]
    h=[x/x.sum() for x in h]
    def safe_corr(a,b):
        try: return float(np.corrcoef(a,b)[0,1])
        except: return 0.0
    return np.array([safe_corr(h[0],h[1]),safe_corr(h[0],h[2]),safe_corr(h[1],h[2])])

def _wavelet_feats(gray):
    g=gray[:gray.shape[0]-(gray.shape[0]%2),:gray.shape[1]-(gray.shape[1]%2)]/255.0
    LH=(g[0::2,0::2]+g[0::2,1::2]-g[1::2,0::2]-g[1::2,1::2])/4.0
    HL=(g[0::2,0::2]-g[0::2,1::2]+g[1::2,0::2]-g[1::2,1::2])/4.0
    HH=(g[0::2,0::2]-g[0::2,1::2]-g[1::2,0::2]+g[1::2,1::2])/4.0
    return np.array([LH.std()*10,HL.std()*10,HH.std()*10])

def _markov_feats(gray_uint8):
    lsb=(gray_uint8&1).astype(int)
    a=lsb[:,:-1].flatten(); b=lsb[:,1:].flatten()
    T=np.zeros((2,2))
    for i in range(2):
        for j in range(2):
            T[i,j]=np.sum((a==i)&(b==j))
    rs=T.sum(axis=1,keepdims=True); rs[rs==0]=1
    return (T/rs).flatten()

def _benford_feats(arr):
    exp=np.array([math.log10(1+1/d) for d in range(1,10)])
    def dev(vals):
        nz=np.abs(vals[vals!=0]).astype(float)
        if len(nz)<50: return 0.0
        log=np.log10(nz); lead=np.floor(10.0**(log-np.floor(log))).astype(int)
        lead=lead[(lead>=1)&(lead<=9)]
        if len(lead)==0: return 0.0
        c=np.bincount(lead,minlength=10)[1:10].astype(float); c/=c.sum()
        return float(np.sum(np.abs(c-exp)))
    pix=dev(arr.flatten().astype(float))
    diff=dev(np.abs(arr[:,1:,:].astype(int)-arr[:,:-1,:].astype(int)).flatten().astype(float))
    return np.array([pix,diff])

def _pvd_feats(gray_uint8):
    arr=gray_uint8.astype(int)
    diffs=np.abs(arr[:,1:]-arr[:,:-1]).flatten()
    pdh=np.bincount(diffs,minlength=256)
    pvd_ranges=[(0,7),(8,15),(16,31),(32,63),(64,127),(128,255)]
    flatness_scores=[]
    for lo,hi in pvd_ranges:
        seg=pdh[lo:hi+1].astype(float)
        if seg.sum()<10: flatness_scores.append(0.0); continue
        mean_v=seg.mean()
        if mean_v<1: flatness_scores.append(0.0); continue
        cv=seg.std()/mean_v
        flatness_scores.append(max(0.0,1.0-min(cv/1.5,1.0)))
    return np.array(flatness_scores)

def _glcm_feats(gray_uint8):
    g=(gray_uint8//8).astype(int)
    G=32
    glcm=np.zeros((G,G),dtype=float)
    a=g[:,:-1].flatten(); b=g[:,1:].flatten()
    for i in range(len(a)): glcm[a[i],b[i]]+=1
    glcm/=glcm.sum()+1e-9
    I,J=np.meshgrid(np.arange(G),np.arange(G),indexing='ij')
    contrast   =float(np.sum(glcm*(I-J)**2))
    energy     =float(np.sum(glcm**2))
    homogeneity=float(np.sum(glcm/(1+(I-J)**2)))
    ent_g      =float(-np.sum(glcm[glcm>0]*np.log2(glcm[glcm>0])))
    mu_i=(I*glcm).sum(); mu_j=(J*glcm).sum()
    sig_i=np.sqrt(((I-mu_i)**2*glcm).sum()+1e-9)
    sig_j=np.sqrt(((J-mu_j)**2*glcm).sum()+1e-9)
    correlation=float(((I-mu_i)*(J-mu_j)*glcm).sum()/(sig_i*sig_j+1e-9))
    av=g[:-1,:].flatten(); bv=g[1:,:].flatten()
    glcm_v=np.zeros((G,G),dtype=float)
    for i in range(len(av)): glcm_v[av[i],bv[i]]+=1
    glcm_v/=glcm_v.sum()+1e-9
    energy_v  =float(np.sum(glcm_v**2))
    contrast_v=float(np.sum(glcm_v*(I-J)**2))
    return np.array([contrast/1000,energy*10,homogeneity,ent_g/10,
                     correlation,energy_v*10,contrast_v/1000,
                     abs(energy-energy_v)*100])

def _ws_feats(arr):
    feats=[]
    weights=np.array([[1,2,1],[2,4,2],[1,2,1]],dtype=float)/16.0
    for ch in range(3):
        plane=arr[:,:,ch].astype(float)
        smoothed=_conv2d(plane,weights,mode='same',boundary='symm')
        residual=plane-smoothed
        lsb=(arr[:,:,ch]&1).astype(float)-0.5
        ws_stat=float(np.sum(lsb*residual))/(plane.size+1e-9)
        feats.append(np.clip(abs(ws_stat)*100,0,1))
    feats.append(np.std([feats[0],feats[1],feats[2]]))
    return np.array(feats[:4])

def _rs_light(arr):
    def rs_asym(ch_arr):
        flat=ch_arr.flatten(); N=len(flat)//4*4; flat=flat[:N]
        groups=flat.reshape(-1,4)
        def smooth(g): return np.sum(np.abs(np.diff(g.astype(float))))
        R=S=Rn=Sn=0; mask=np.array([0,1,0,1])
        for g in groups[:500]:
            f0=smooth(g)
            g_f=(g.copy()); g_f[mask==1]^=1
            f1=smooth(g_f)
            g_n=g.copy()
            for i in np.where(mask)[0]:
                v=int(g_n[i])
                g_n[i]=255 if v==0 else (v-1 if v%2==1 else v+1)
            fn=smooth(g_n)
            if f1>f0: R+=1
            elif f1<f0: S+=1
            if fn>f0: Rn+=1
            elif fn<f0: Sn+=1
        tot=max(500,1)
        return abs(R/tot-Rn/tot)
    feats=[]
    for ch in range(3): feats.append(rs_asym(arr[:,:,ch]))
    feats.append(max(feats))
    feats.append(np.mean(feats[:3]))
    feats.append(np.std(feats[:3]))
    return np.array(feats[:6])

def _calibration_feats(arr):
    try:
        from PIL import Image as _Img
        buf=io.BytesIO()
        _Img.fromarray(arr.astype(np.uint8)).save(buf,format='JPEG',quality=75)
        buf.seek(0)
        recomp=np.array(_Img.open(buf).convert('RGB')).astype(float)
        diff=np.abs(arr.astype(float)-recomp)
        return np.array([diff.mean()/255,diff.std()/255,diff.max()/255])
    except Exception:
        return np.zeros(3)

def _lsb_4gram(gray_uint8, max_n=50000):
    lsb=(gray_uint8.flatten()&1).astype(int)
    if len(lsb)>max_n: lsb=lsb[:max_n]
    counts=np.zeros(16,dtype=float)
    for i in range(len(lsb)-3):
        idx=lsb[i]*8+lsb[i+1]*4+lsb[i+2]*2+lsb[i+3]
        counts[idx]+=1
    counts/=counts.sum()+1e-9
    return counts

def extract_all_features(img):
    arr=img.astype(np.uint8)
    if arr.ndim==2: arr=np.stack([arr]*3,axis=-1)
    gray=_gray(arr); gray01=gray/255.0; gray_u8=gray.astype(np.uint8)
    parts=[
        _srm_features(gray01),
        _srm_rgb(arr),
        _lsb_entropy_feats(arr),
        _chi_square_feats(arr),
        _moments_feats(arr),
        _gradient_feats(gray),
        _fft_feats(gray),
        _rle_feats(gray_u8),
        _color_corr_feats(arr),
        _wavelet_feats(gray),
        _markov_feats(gray_u8),
        _benford_feats(arr),
        _pvd_feats(gray_u8),
        _glcm_feats(gray_u8),
        _ws_feats(arr),
        _rs_light(arr),
        _calibration_feats(arr),
        _lsb_4gram(gray_u8),
    ]
    fv=np.concatenate([np.nan_to_num(p,nan=0.0,posinf=0.0,neginf=0.0) for p in parts])
    if len(fv)<256: fv=np.pad(fv,(0,256-len(fv)))
    else: fv=fv[:256]
    return fv.astype(np.float64)

# ── Model Loader ─────────────────────────────────────────────────────────────
class ModelWrapper:
    def __init__(self):
        self.robust_svm = None
        self.urd = None
        self.loaded = False

    _RELEASE_BASE = 'https://github.com/ahmedA-gif/cv-project-stegno-analysis/releases/download/v1.0.0'

    def _download_if_missing(self, path, url):
        if os.path.exists(path):
            return True
        print(f"  Downloading {os.path.basename(path)} from GitHub release...")
        try:
            urllib.request.urlretrieve(url, path)
            size_mb = os.path.getsize(path) / (1024*1024)
            print(f"  Downloaded {size_mb:.0f} MB")
            return True
        except Exception as e:
            print(f"  Download failed: {e}")
            return False

    def load(self):
        svm_path = os.path.join(MODELS_DIR, 'robust_svm.pkl')
        urd_path = os.path.join(MODELS_DIR, 'urd.pkl')

        self._download_if_missing(svm_path, f'{self._RELEASE_BASE}/robust_svm.pkl')
        self._download_if_missing(urd_path, f'{self._RELEASE_BASE}/urd.pkl')

        if os.path.exists(svm_path):
            try:
                d = joblib.load(svm_path)
                self.robust_svm = d
                print(f"  robust_svm.pkl loaded (threshold={d.get('best_threshold',0.5):.3f})")
            except Exception as e:
                print(f"  Failed to load robust_svm.pkl: {e}")

        if os.path.exists(urd_path):
            try:
                d = joblib.load(urd_path)
                self.urd = d
                print(f"  urd.pkl loaded (threshold={d.get('best_threshold',0.5):.3f})")
            except Exception as e:
                print(f"  Failed to load urd.pkl: {e}")

        self.loaded = (self.robust_svm is not None or self.urd is not None)
        return self.loaded

    def predict_svm(self, fv):
        if self.robust_svm is None:
            return 0.5, 0.5
        svm = self.robust_svm['svm']
        scaler = self.robust_svm['scaler']
        pca = self.robust_svm['pca']
        threshold = self.robust_svm.get('best_threshold', 0.5)
        Xs = scaler.transform(fv.reshape(1, -1))
        Xp = pca.transform(Xs)
        prob = svm.predict_proba(Xp)[0, 1]
        return float(prob), float(threshold)

    def predict_urd(self, fv):
        if self.urd is None:
            return 0.5, 0.5
        base_models = self.urd['base_models']
        meta_model = self.urd['meta_model']
        scalers = self.urd['scalers']
        threshold = self.urd.get('best_threshold', 0.5)

        svm_scaler = scalers['svm']
        pca_svm = scalers['pca_svm']
        tree_scaler = scalers['tree']

        Xs_svm = pca_svm.transform(svm_scaler.transform(fv.reshape(1, -1)))
        Xs_tree = tree_scaler.transform(fv.reshape(1, -1))

        preds = []
        for mname, model in base_models.items():
            Xs = Xs_svm if mname == 'svm' else Xs_tree
            preds.append(model.predict_proba(Xs)[:, 1])
        oof = np.column_stack(preds)
        prob = meta_model.predict_proba(oof)[0, 1]
        return float(prob), float(threshold)

    def predict_image(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            return None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if max(img.shape[:2]) > 512:
            img_rgb = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_AREA)
        fv = extract_all_features(img_rgb)
        results = {}
        if self.robust_svm:
            prob_svm, thr_svm = self.predict_svm(fv)
            results['robust_svm'] = {
                'probability': round(prob_svm, 4),
                'prediction': 'STEGO' if prob_svm > thr_svm else 'CLEAN',
                'threshold': round(thr_svm, 3),
                'confidence': round(abs(prob_svm - 0.5) * 200, 1)
            }
        if self.urd:
            prob_urd, thr_urd = self.predict_urd(fv)
            results['urd'] = {
                'probability': round(prob_urd, 4),
                'prediction': 'STEGO' if prob_urd > thr_urd else 'CLEAN',
                'threshold': round(thr_urd, 3),
                'confidence': round(abs(prob_urd - 0.5) * 200, 1)
            }
        return results

model_wrapper = ModelWrapper()
model_wrapper.load()

# ── YOLO Pool (lazy, multi-model) ────────────────────────────────────────────
from modules.yolo_detector import YOLOPool
yolo_pool = YOLOPool()
stego_analyzer = None

def get_yolo(model_name=None):
    return yolo_pool.get_model(model_name)

def get_analyzer():
    global stego_analyzer
    if stego_analyzer is None:
        from modules.stego_analyzer import StegoAnalyzer
        yd = get_yolo()
        if yd:
            stego_analyzer = StegoAnalyzer(model_wrapper=model_wrapper, yolo_detector=yd)
            print(f"  StegoAnalyzer loaded")
    return stego_analyzer

# ── Forensic Modules (lazy) ──────────────────────────────────────────────────
_ela_analyzer = None
_text_detector = None
_gan_detector = None

def get_ela():
    global _ela_analyzer
    if _ela_analyzer is None:
        from modules.ela_forensics import ELAAnalyzer
        from modules.config import ELA_QUALITY, ELA_SCALE
        _ela_analyzer = ELAAnalyzer(quality=ELA_QUALITY, scale=ELA_SCALE)
    return _ela_analyzer

def get_ocr():
    global _text_detector
    if _text_detector is None:
        from modules.text_ocr import TextDetector
        from modules.config import OCR_LANGUAGES, OCR_GPU
        _text_detector = TextDetector(languages=OCR_LANGUAGES, gpu=OCR_GPU)
    return _text_detector

def get_gan():
    global _gan_detector
    if _gan_detector is None:
        from modules.gan_detector import GANDetector
        _gan_detector = GANDetector()
    return _gan_detector

# ── Tracker State ────────────────────────────────────────────────────────────
_active_tracker_name = "bytetrack"
_active_tracker = None

def get_tracker(name=None):
    global _active_tracker, _active_tracker_name
    name = (name or _active_tracker_name).lower().strip()
    yd = get_yolo()
    if yd is None or yd.model is None:
        return None
    if _active_tracker is None or _active_tracker_name != name:
        from modules.trackers.tracker_factory import get_tracker as _factory, list_trackers
        if name not in list_trackers():
            name = "bytetrack"
        _active_tracker_name = name
        _active_tracker = _factory(name, yd.model)
        print(f"  Tracker switched to {_active_tracker_name}")
    return _active_tracker

# ── Flask Routes ─────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def api_status():
    return jsonify({
        'status': 'online',
        'models_loaded': model_wrapper.loaded,
        'robust_svm': model_wrapper.robust_svm is not None,
        'urd': model_wrapper.urd is not None,
        'version': '4.2.0',
        'feature_dim': 256,
    })

@app.route('/api/metrics')
def api_metrics():
    active_model = 'urd' if model_wrapper.urd else 'robust_svm' if model_wrapper.robust_svm else None
    return jsonify({
        'active_model': active_model,
        'models': VALIDATION_METRICS,
        'robust_svm_loaded': model_wrapper.robust_svm is not None,
        'urd_loaded': model_wrapper.urd is not None,
    })

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if not model_wrapper.loaded:
        return jsonify({'error': 'Models not loaded - server still initializing'}), 503
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    fname = f"{int(time.time())}_{file.filename}"
    fpath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    file.save(fpath)
    try:
        t0 = time.time()
        results = model_wrapper.predict_image(fpath)
        elapsed = round(time.time() - t0, 3)
        if results is None:
            return jsonify({'error': 'Could not read image'}), 400
        return jsonify({
            'success': True,
            'filename': file.filename,
            'inference_time': elapsed,
            'results': results,
            'feature_dim': 256,
        })
    finally:
        try:
            os.remove(fpath)
        except:
            pass

# ── Tracker Routes ─────────────────────────────────────────────────────────
@app.route('/api/tracker/list')
def api_tracker_list():
    from modules.trackers.tracker_factory import list_trackers
    return jsonify({
        'trackers': list_trackers(),
        'active': _active_tracker_name,
    })

@app.route('/api/tracker/select', methods=['POST'])
def api_tracker_select():
    name = request.json.get('tracker', 'bytetrack') if request.is_json else request.form.get('tracker', 'bytetrack')
    from modules.trackers.tracker_factory import list_trackers
    if name not in list_trackers():
        return jsonify({'error': f"Unknown tracker '{name}'", 'available': list_trackers()}), 400
    t = get_tracker(name)
    if t is None:
        return jsonify({'error': 'YOLO model not loaded'}), 503
    return jsonify({'success': True, 'active_tracker': _active_tracker_name, 'available': list_trackers()})

@app.route('/api/tracker/track', methods=['POST'])
def api_tracker_track():
    tr = get_tracker()
    if tr is None:
        return jsonify({'error': 'No tracker available'}), 503
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    file = request.files['image']
    fname = f"{int(time.time())}_{file.filename}"
    fpath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    file.save(fpath)
    try:
        yd = get_yolo()
        results = yd.detect(fpath)
        img = cv2.imread(fpath)
        from modules.utils import image_to_base64, draw_detections
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            annotated = draw_detections(img_rgb, [{'box': d['box'], 'label': d['label'], 'confidence': d['confidence']} for d in results])
            annotated_b64 = 'data:image/png;base64,' + image_to_base64(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        else:
            annotated_b64 = None
        return jsonify({
            'success': True,
            'tracker': _active_tracker_name,
            'num_detections': len(results),
            'detections': serialize(results),
            'annotated_image': annotated_b64,
        })
    finally:
        try: os.remove(fpath)
        except: pass

# ── YOLO Detection Routes ──────────────────────────────────────────────────
@app.route('/api/yolo/status')
def api_yolo_status():
    yd = get_yolo()
    if yd is None or yd.model is None:
        return jsonify({'available': False, 'status': 'not loaded'})
    info = yd.get_model_info()
    return jsonify({'available': True, 'status': 'loaded', 'info': serialize(info)})

@app.route('/api/yolo/detect', methods=['POST'])
def api_yolo_detect():
    yd = get_yolo()
    if yd is None or yd.model is None:
        return jsonify({'error': 'YOLO model not loaded'}), 503
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    file = request.files['image']
    fname = f"{int(time.time())}_{file.filename}"
    fpath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    file.save(fpath)
    try:
        from modules.utils import image_to_base64, draw_detections
        from modules.residual_heatmap import ResidualHeatmap
        img = cv2.imread(fpath)
        if img is None:
            return jsonify({'error': 'Could not read image'}), 400
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hm = ResidualHeatmap()
        srm = hm.compute_srm_energy(img_rgb)
        heatmap = hm.generate_heatmap(img_rgb)
        detections = yd.detect_stego_regions(img_rgb, srm)
        annotated = draw_detections(img_rgb, detections)
        heatmap_overlay = hm.overlay_heatmap(img_rgb, heatmap)
        return jsonify({
            'success': True,
            'num_detections': len(detections),
            'detections': serialize(detections),
            'annotated_image': 'data:image/png;base64,' + image_to_base64(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)),
            'heatmap_image': 'data:image/png;base64,' + image_to_base64(cv2.cvtColor(heatmap_overlay, cv2.COLOR_RGB2BGR)),
        })
    finally:
        try: os.remove(fpath)
        except: pass

@app.route('/api/yolo/analyze', methods=['POST'])
def api_yolo_analyze():
    yd = get_yolo()
    if yd is None or yd.model is None:
        return jsonify({'error': 'YOLO model not loaded'}), 503
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    file = request.files['image']
    fname = f"{int(time.time())}_{file.filename}"
    fpath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    file.save(fpath)
    try:
        img = cv2.imread(fpath)
        if img is None:
            return jsonify({'error': 'Could not read image'}), 400
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if max(img_rgb.shape[:2]) > 1024:
            scale = 1024 / max(img_rgb.shape[:2])
            nw, nh = int(img_rgb.shape[1]*scale), int(img_rgb.shape[0]*scale)
            img_rgb = cv2.resize(img_rgb, (nw, nh))
        from modules.residual_heatmap import ResidualHeatmap
        hm = ResidualHeatmap()
        srm = hm.compute_srm_energy(img_rgb)
        heatmap = hm.generate_heatmap(img_rgb)
        detections = yd.detect_stego_regions(img_rgb, srm)
        if model_wrapper.loaded and detections:
            for det in detections:
                x1, y1, x2, y2 = map(int, det["box"])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_rgb.shape[1], x2), min(img_rgb.shape[0], y2)
                if x2 > x1 and y2 > y1:
                    roi = img_rgb[y1:y2, x1:x2]
                    if max(roi.shape[:2]) > 512:
                        scale = 512 / max(roi.shape[:2])
                        roi = cv2.resize(roi, (int(roi.shape[1]*scale), int(roi.shape[0]*scale)))
                    fv = extract_all_features(roi)
                    svm_prob, _ = model_wrapper.predict_svm(fv)
                    urd_prob, _ = model_wrapper.predict_urd(fv)
                    det["stego_prob"] = round(max(svm_prob, urd_prob), 4)
                    det["svm_prob"] = round(svm_prob, 4)
                    det["urd_prob"] = round(urd_prob, 4)
        hotspots = hm.get_stego_hotspots(heatmap)
        from modules.utils import draw_detections
        annotated = draw_detections(img_rgb, detections)
        heatmap_overlay = hm.overlay_heatmap(img_rgb, heatmap)
        return jsonify({
            'success': True,
            'num_detections': len(detections),
            'detections': serialize(detections),
            'num_hotspots': len(hotspots),
            'hotspots': serialize(hotspots),
            'annotated_image': 'data:image/png;base64,' + image_to_base64(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)),
            'heatmap_image': 'data:image/png;base64,' + image_to_base64(cv2.cvtColor(heatmap_overlay, cv2.COLOR_RGB2BGR)),
        })
    finally:
        try: os.remove(fpath)
        except: pass

# ── YOLO Model Switching ────────────────────────────────────────────────────
@app.route('/api/yolo/models')
def api_yolo_models():
    from modules.yolo_detector import YOLOPool
    return jsonify({
        'models': YOLOPool.list_models(),
        'active': yolo_pool.active_name,
    })

@app.route('/api/yolo/switch', methods=['POST'])
def api_yolo_switch():
    data = request.get_json(silent=True) or {}
    name = data.get('model', '').strip().lower()
    if not name:
        return jsonify({'error': 'No model name provided'}), 400
    ok, msg = yolo_pool.switch(name)
    if not ok:
        return jsonify({'error': msg}), 400
    return jsonify({'success': True, 'model': yolo_pool.active_name, 'message': msg})

# ── Forensic Routes ─────────────────────────────────────────────────────────
@app.route('/api/forensics/ela', methods=['POST'])
def api_forensics_ela():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    file = request.files['image']
    fname = f"{int(time.time())}_{file.filename}"
    fpath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    file.save(fpath)
    try:
        img = cv2.imread(fpath)
        if img is None:
            return jsonify({'error': 'Could not read image'}), 400
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if max(img_rgb.shape[:2]) > 1024:
            scale = 1024 / max(img_rgb.shape[:2])
            nw, nh = int(img_rgb.shape[1] * scale), int(img_rgb.shape[0] * scale)
            img_rgb = cv2.resize(img_rgb, (nw, nh))
        ela = get_ela()
        from modules.utils import image_to_base64
        result = ela.analyze(img_rgb)
        overlay = ela.overlay_ela(img_rgb)
        result['ela_image'] = 'data:image/png;base64,' + image_to_base64(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        return jsonify({'success': True, 'ela': serialize(result)})
    finally:
        try: os.remove(fpath)
        except: pass

@app.route('/api/forensics/ocr', methods=['POST'])
def api_forensics_ocr():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    file = request.files['image']
    fname = f"{int(time.time())}_{file.filename}"
    fpath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    file.save(fpath)
    try:
        img = cv2.imread(fpath)
        if img is None:
            return jsonify({'error': 'Could not read image'}), 400
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if max(img_rgb.shape[:2]) > 1024:
            scale = 1024 / max(img_rgb.shape[:2])
            img_rgb = cv2.resize(img_rgb, (int(img_rgb.shape[1] * scale), int(img_rgb.shape[0] * scale)))
        ocr = get_ocr()
        from modules.utils import image_to_base64
        result = ocr.analyze(img_rgb)
        annotated = ocr.draw_text_regions(img_rgb, result['detections'])
        result['annotated_image'] = 'data:image/png;base64,' + image_to_base64(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        return jsonify({'success': True, 'ocr': serialize(result)})
    finally:
        try: os.remove(fpath)
        except: pass

@app.route('/api/forensics/gan', methods=['POST'])
def api_forensics_gan():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    file = request.files['image']
    fname = f"{int(time.time())}_{file.filename}"
    fpath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    file.save(fpath)
    try:
        img = cv2.imread(fpath)
        if img is None:
            return jsonify({'error': 'Could not read image'}), 400
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if max(img_rgb.shape[:2]) > 1024:
            scale = 1024 / max(img_rgb.shape[:2])
            img_rgb = cv2.resize(img_rgb, (int(img_rgb.shape[1] * scale), int(img_rgb.shape[0] * scale)))
        gan = get_gan()
        result = gan.predict(img_rgb)
        return jsonify({'success': True, 'gan': serialize(result)})
    finally:
        try: os.remove(fpath)
        except: pass

@app.route('/api/forensics/full', methods=['POST'])
def api_forensics_full():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    file = request.files['image']
    fname = f"{int(time.time())}_{file.filename}"
    fpath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    file.save(fpath)
    try:
        img = cv2.imread(fpath)
        if img is None:
            return jsonify({'error': 'Could not read image'}), 400
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if max(img_rgb.shape[:2]) > 1024:
            scale = 1024 / max(img_rgb.shape[:2])
            img_rgb = cv2.resize(img_rgb, (int(img_rgb.shape[1] * scale), int(img_rgb.shape[0] * scale)))
        t0 = time.time()
        results = {}
        # 1 — Steganalysis (256-dim)
        if model_wrapper.loaded:
            fv = extract_all_features(img_rgb)
            if model_wrapper.robust_svm:
                p, t = model_wrapper.predict_svm(fv)
                results['steganalysis_svm'] = {'probability': round(p, 4), 'prediction': 'STEGO' if p > t else 'CLEAN'}
            if model_wrapper.urd:
                p, t = model_wrapper.predict_urd(fv)
                results['steganalysis_urd'] = {'probability': round(p, 4), 'prediction': 'STEGO' if p > t else 'CLEAN'}
        # 2 — YOLO detection
        yd = get_yolo()
        if yd and yd.model:
            from modules.residual_heatmap import ResidualHeatmap
            hm = ResidualHeatmap()
            srm = hm.compute_srm_energy(img_rgb)
            dets = yd.detect_stego_regions(img_rgb, srm)
            if dets:
                for det in dets:
                    x1, y1, x2, y2 = map(int, det["box"])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(img_rgb.shape[1], x2), min(img_rgb.shape[0], y2)
                    if x2 > x1 and y2 > y1 and model_wrapper.loaded:
                        roi = img_rgb[y1:y2, x1:x2]
                        if max(roi.shape[:2]) > 512:
                            sc = 512 / max(roi.shape[:2])
                            roi = cv2.resize(roi, (int(roi.shape[1] * sc), int(roi.shape[0] * sc)))
                        fv_roi = extract_all_features(roi)
                        svm_p, _ = model_wrapper.predict_svm(fv_roi)
                        urd_p, _ = model_wrapper.predict_urd(fv_roi)
                        det['stego_prob'] = round(max(svm_p, urd_p), 4)
                        det['svm_prob'] = round(svm_p, 4)
                        det['urd_prob'] = round(urd_p, 4)
            results['yolo_detections'] = dets
        # 3 — ELA
        try:
            ela = get_ela()
            results['ela'] = ela.analyze(img_rgb)
        except Exception as e:
            results['ela'] = {'error': str(e)}
        # 4 — OCR
        try:
            ocr = get_ocr()
            ocr_res = ocr.analyze(img_rgb)
            results['ocr'] = {'num_text_regions': ocr_res['num_text_regions'], 'total_characters': ocr_res['total_characters']}
        except Exception as e:
            results['ocr'] = {'error': str(e)}
        # 5 — GAN
        try:
            gan = get_gan()
            results['gan'] = gan.predict(img_rgb)
        except Exception as e:
            results['gan'] = {'error': str(e)}
        elapsed = round(time.time() - t0, 3)
        return jsonify({'success': True, 'inference_time': elapsed, 'results': serialize(results)})
    finally:
        try: os.remove(fpath)
        except: pass

# ── Animated Metrics (GIF) ──────────────────────────────────────────────────
@app.route('/api/metrics/animated')
def api_metrics_animated():
    import matplotlib.pyplot as plt
    import io as _io
    from matplotlib.animation import FuncAnimation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor('#0e0e0e')
    for ax in [ax1, ax2]:
        ax.set_facecolor('#0e0e0e')
        ax.tick_params(colors='#e5e2e1')
        for spine in ax.spines.values(): spine.set_color('#3b4b37')

    data = VALIDATION_METRICS['urd']['per_dataset']
    names = list(data.keys())
    aucs = [data[n]['auc'] for n in names]
    accs = [data[n]['acc'] for n in names]
    x = np.arange(len(names))

    bars1 = ax1.bar(x, [0]*len(names), color='#00e639', alpha=0.8, label='AUC')
    bars2 = ax2.bar(x, [0]*len(names), color='#00eefc', alpha=0.8, label='Accuracy')
    ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=7, color='#e5e2e1')
    ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=7, color='#e5e2e1')
    ax1.legend(); ax2.legend()
    ax1.set_ylabel('AUC', color='#e5e2e1'); ax2.set_ylabel('Accuracy', color='#e5e2e1')
    ax1.set_ylim(0, 1.1); ax2.set_ylim(0, 1.1)

    def animate(frame):
        progress = min(1.0, (frame + 1) / 30)
        for i in range(len(names)):
            bars1[i].set_height(aucs[i] * progress)
            bars2[i].set_height(accs[i] * progress)
        return bars1 + bars2

    anim = FuncAnimation(fig, animate, frames=30, interval=50, blit=True)
    buf = _io.BytesIO()
    anim.save(buf, writer='pillow', fps=20, dpi=80)
    buf.seek(0)
    plt.close(fig)
    return send_file(buf, mimetype='image/gif')

# ── Helper ──────────────────────────────────────────────────────────────────
def serialize(obj):
    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [serialize(v) for v in obj]
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj

# ── Save histogram for initial metrics display ──────────────────────────────
@app.route('/api/metrics/histogram')
def api_metrics_histogram():
    import matplotlib.pyplot as plt
    import io as _io
    fig, ax = plt.subplots(figsize=(8, 4))
    model = 'urd'
    data = VALIDATION_METRICS[model]['per_dataset']
    names = list(data.keys())
    aucs = [data[n]['auc'] for n in names]
    accs = [data[n]['acc'] for n in names]
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w/2, aucs, w, label='AUC', color='#00e639', alpha=0.8)
    ax.bar(x + w/2, accs, w, label='Accuracy', color='#00eefc', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8, color='#e5e2e1')
    ax.set_ylabel('Score', color='#e5e2e1')
    ax.legend()
    ax.set_facecolor('#0e0e0e')
    fig.patch.set_facecolor('#0e0e0e')
    ax.tick_params(colors='#e5e2e1')
    for spine in ax.spines.values(): spine.set_color('#3b4b37')
    buf = _io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#0e0e0e')
    buf.seek(0)
    plt.close(fig)
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    print(f"Starting development server on http://0.0.0.0:{port}")
    app.run(debug=False, host='0.0.0.0', port=port)
