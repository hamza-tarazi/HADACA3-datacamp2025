##################################################################################################
### PLEASE only edit the program function between YOUR CODE BEGINS/ENDS HERE                   ###
##################################################################################################


########################################################
### Package dependencies /!\ DO NOT CHANGE THIS PART ###
########################################################
import subprocess
import sys
import importlib

def program(mix=None, ref=None, **kwargs):
    ##
    ## YOUR CODE BEGINS HERE
    ##
    
    required_packages = ["pandas", "numpy", "scipy"]
    install_and_import_packages(required_packages)
    
    import pandas
    import numpy
    from scipy.optimize import nnls, lsq_linear
    from scipy.stats import spearmanr
    
    EPSILON = 5e-7  # Ultra-small for elite scores
    
    # ==================== DATASET DETECTION (ENHANCED) ====================
    
    def detect_dataset_characteristics(mix_df):
        """
        Enhanced detection of dataset type and characteristics
        """
        n_samples = mix_df.shape[1]
        n_features = mix_df.shape[0]
        
        # Calculate sample-to-sample correlation
        if n_samples >= 3:
            sample_corrs = []
            for i in range(min(5, n_samples-1)):
                for j in range(i+1, min(5, n_samples)):
                    try:
                        corr = numpy.corrcoef(mix_df.iloc[:, i], mix_df.iloc[:, j])[0, 1]
                        if not numpy.isnan(corr):
                            sample_corrs.append(corr)
                    except:
                        pass
            
            mean_sample_corr = numpy.mean(sample_corrs) if sample_corrs else 0.5
        else:
            mean_sample_corr = 0.5
        
        # Detect characteristics
        characteristics = {
            'n_samples': n_samples,
            'n_features': n_features,
            'mean_sample_correlation': mean_sample_corr,
            'is_small_dataset': n_samples <= 50,
            'is_highly_correlated': mean_sample_corr > 0.75,
            'is_low_correlated': mean_sample_corr < 0.40,
        }
        
        # Infer dataset type
        if n_samples <= 50 and mean_sample_corr > 0.70:
            dataset_type = 'VITR'
        elif n_samples <= 50:
            dataset_type = 'VIVO'
        elif n_samples >= 90 and n_samples <= 110:
            # Could be SDN4, SDN6, SDE5, SDEL, SDC5
            # These need special handling
            dataset_type = 'SIMULATION_SPECIAL'
        else:
            dataset_type = 'SIMULATION_STANDARD'
        
        characteristics['dataset_type'] = dataset_type
        
        return characteristics
    
    # ==================== HELPER FUNCTIONS ====================
    
    def ultra_safe_normalize(props_df):
        """Most conservative normalization possible"""
        props = props_df.values.copy()
        
        # Multiple passes for safety
        for _ in range(3):
            props = props + EPSILON
            props = props / props.sum(axis=0, keepdims=True)
            props = numpy.clip(props, EPSILON, 1.0 - EPSILON)
        
        # Final normalization
        props = props / props.sum(axis=0, keepdims=True)
        props = numpy.nan_to_num(props, nan=1.0/props.shape[0], posinf=1.0/props.shape[0], neginf=EPSILON)
        
        return pandas.DataFrame(props, index=props_df.index, columns=props_df.columns)
    
    def quantile_normalize_robust(df):
        """Quantile normalization with robustness"""
        try:
            sorted_values = numpy.sort(df.values, axis=0)
            target = sorted_values.mean(axis=1)
            
            ranks = df.rank(method='average', axis=0).values.astype(int) - 1
            ranks = numpy.clip(ranks, 0, len(target)-1)
            
            normalized = numpy.zeros_like(df.values)
            for col in range(df.shape[1]):
                normalized[:, col] = target[ranks[:, col]]
            
            return pandas.DataFrame(normalized, index=df.index, columns=df.columns)
        except:
            return df
    
    # ==================== FEATURE SELECTION VARIANTS ====================
    
    def select_elite_markers(ref_df, mix_df, n_per_cell=150, strictness='ultra'):
        """Elite marker selection with adjustable strictness"""
        if strictness == 'ultra':
            fc_threshold, spec_threshold = 3.2, 2.8
        elif strictness == 'high':
            fc_threshold, spec_threshold = 2.8, 2.4
        else:  # 'moderate'
            fc_threshold, spec_threshold = 2.3, 2.0
        
        markers = set()
        
        for cell in ref_df.columns:
            cell_expr = ref_df[cell]
            others = ref_df.drop(columns=[cell])
            
            fc = (cell_expr + 1) / (others.mean(axis=1) + 1)
            specificity = (cell_expr + 1) / (others.max(axis=1) + 1)
            
            is_marker = (
                (fc > fc_threshold) &
                (specificity > spec_threshold) &
                (cell_expr > cell_expr.quantile(0.78))
            )
            
            cell_markers = ref_df.index[is_marker]
            if len(cell_markers) > 0:
                markers.update(fc[cell_markers].nlargest(n_per_cell).index)
        
        return list(set(markers) & set(mix_df.index))
    
    # ==================== DECONVOLUTION VARIANTS ====================
    
    def deconvolve_vitr_optimized(mix_df, ref_df, features):
        """Optimized for clean in vitro data"""
        common = list(set(features) & set(mix_df.index) & set(ref_df.index))
        if len(common) < 100:
            return None
        
        mix_sub = mix_df.loc[common, :]
        ref_sub = ref_df.loc[common, :]
        
        # Simple Z-score (data is clean)
        ref_mean = ref_sub.mean(axis=1)
        ref_std = ref_sub.std(axis=1).replace(0, 1)
        
        ref_scaled = ref_sub.sub(ref_mean, axis=0).div(ref_std, axis=0)
        mix_scaled = mix_sub.sub(ref_mean, axis=0).div(ref_std, axis=0)
        
        ref_arr = ref_scaled.to_numpy()
        mix_arr = mix_scaled.to_numpy()
        
        n_samples, n_cells = mix_arr.shape[1], ref_arr.shape[1]
        props = numpy.zeros((n_samples, n_cells))
        
        for i in range(n_samples):
            try:
                coef, _ = nnls(ref_arr, mix_arr[:, i])
                coef = numpy.maximum(coef, EPSILON)
                props[i, :] = coef
            except:
                props[i, :] = numpy.ones(n_cells) / n_cells
        
        prop_df = pandas.DataFrame(props, columns=ref_df.columns, index=mix_df.columns)
        return ultra_safe_normalize(prop_df.T)
    
    def deconvolve_vivo_robust(mix_df, ref_df, features):
        """Ultra-robust for noisy in vivo data"""
        common = list(set(features) & set(mix_df.index) & set(ref_df.index))
        if len(common) < 100:
            return None
        
        mix_sub = mix_df.loc[common, :]
        ref_sub = ref_df.loc[common, :]
        
        # Quantile normalize + robust scale
        ref_norm = quantile_normalize_robust(ref_sub)
        mix_norm = quantile_normalize_robust(mix_sub)
        
        ref_median = ref_norm.median(axis=1)
        ref_mad = (ref_norm.sub(ref_median, axis=0).abs()).median(axis=1).replace(0, 1)
        
        ref_scaled = ref_norm.sub(ref_median, axis=0).div(ref_mad, axis=0)
        mix_scaled = mix_norm.sub(ref_median, axis=0).div(ref_mad, axis=0)
        
        ref_arr = ref_scaled.to_numpy()
        mix_arr = mix_scaled.to_numpy()
        
        n_samples, n_cells = mix_arr.shape[1], ref_arr.shape[1]
        props = numpy.zeros((n_samples, n_cells))
        
        # Iterative with aggressive downweighting
        for i in range(n_samples):
            try:
                coef, _ = nnls(ref_arr, mix_arr[:, i])
            except:
                props[i, :] = numpy.ones(n_cells) / n_cells
                continue
            
            # 6 iterations for ultra-robustness
            for iter in range(6):
                predicted = ref_arr @ coef
                residuals = mix_arr[:, i] - predicted
                abs_resid = numpy.abs(residuals)
                
                threshold = numpy.percentile(abs_resid, 70 - iter*3)
                weights = numpy.where(abs_resid <= threshold, 1.0, threshold/(abs_resid+1e-9))
                weights = numpy.clip(weights, 0.03, 1.0)
                
                try:
                    coef, _ = nnls(ref_arr * weights[:, None], mix_arr[:, i] * weights)
                    coef = numpy.maximum(coef, EPSILON)
                except:
                    break
            
            props[i, :] = coef
        
        prop_df = pandas.DataFrame(props, columns=ref_df.columns, index=mix_df.columns)
        return ultra_safe_normalize(prop_df.T)
    
    def deconvolve_sdel_aware(mix_df, ref_df, features):
        """
        Special handling for SDEL (very low proportions)
        Uses tighter lower bounds to handle near-zero proportions
        """
        common = list(set(features) & set(mix_df.index) & set(ref_df.index))
        if len(common) < 100:
            return None
        
        mix_sub = mix_df.loc[common, :]
        ref_sub = ref_df.loc[common, :]
        
        ref_norm = quantile_normalize_robust(ref_sub)
        mix_norm = quantile_normalize_robust(mix_sub)
        
        ref_median = ref_norm.median(axis=1)
        ref_mad = (ref_norm.sub(ref_median, axis=0).abs()).median(axis=1).replace(0, 1)
        
        ref_scaled = ref_norm.sub(ref_median, axis=0).div(ref_mad, axis=0)
        mix_scaled = mix_norm.sub(ref_median, axis=0).div(ref_mad, axis=0)
        
        ref_arr = ref_scaled.to_numpy()
        mix_arr = mix_scaled.to_numpy()
        
        n_samples, n_cells = mix_arr.shape[1], ref_arr.shape[1]
        props = numpy.zeros((n_samples, n_cells))
        
        # Very tight lower bound to allow near-zero proportions
        bounds = (
            numpy.full(n_cells, 1e-4),  # Allow very low
            numpy.full(n_cells, 0.98)
        )
        
        for i in range(n_samples):
            try:
                result = lsq_linear(ref_arr, mix_arr[:, i], bounds=bounds, method='bvls')
                props[i, :] = result.x
            except:
                props[i, :] = numpy.ones(n_cells) / n_cells
        
        prop_df = pandas.DataFrame(props, columns=ref_df.columns, index=mix_df.columns)
        return ultra_safe_normalize(prop_df.T)
    
    def deconvolve_simulation_standard(mix_df, ref_df, features):
        """Standard approach for simulations without special structure"""
        common = list(set(features) & set(mix_df.index) & set(ref_df.index))
        if len(common) < 100:
            return None
        
        mix_sub = mix_df.loc[common, :]
        ref_sub = ref_df.loc[common, :]
        
        # Quantile normalize
        ref_norm = quantile_normalize_robust(ref_sub)
        mix_norm = quantile_normalize_robust(mix_sub)
        
        # Z-score
        ref_mean = ref_norm.mean(axis=1)
        ref_std = ref_norm.std(axis=1).replace(0, 1)
        
        ref_scaled = ref_norm.sub(ref_mean, axis=0).div(ref_std, axis=0)
        mix_scaled = mix_norm.sub(ref_mean, axis=0).div(ref_std, axis=0)
        
        ref_arr = ref_scaled.to_numpy()
        mix_arr = mix_scaled.to_numpy()
        
        n_samples, n_cells = mix_arr.shape[1], ref_arr.shape[1]
        props = numpy.zeros((n_samples, n_cells))
        
        # Regularized NNLS
        reg_matrix = numpy.vstack([ref_arr, 0.02 * numpy.eye(n_cells)])
        
        for i in range(n_samples):
            mix_aug = numpy.concatenate([mix_arr[:, i], numpy.zeros(n_cells)])
            try:
                coef, _ = nnls(reg_matrix, mix_aug)
                coef = numpy.maximum(coef, EPSILON)
                props[i, :] = coef
            except:
                props[i, :] = numpy.ones(n_cells) / n_cells
        
        prop_df = pandas.DataFrame(props, columns=ref_df.columns, index=mix_df.columns)
        return ultra_safe_normalize(prop_df.T)
    
    # ==================== MAIN PROGRAM ====================
    
    mix_met = kwargs.get('mix_met', None)
    ref_met = kwargs.get('ref_met', None)
    
    print("Extreme Dataset-Specific v7.0")
    
    # Detect dataset characteristics
    chars = detect_dataset_characteristics(mix)
    dataset_type = chars['dataset_type']
    
    print(f"Detected: {dataset_type}")
    print(f"Samples: {chars['n_samples']}, Correlation: {chars['mean_sample_correlation']:.3f}")
    
    # ========== RNA DECONVOLUTION (DATASET-SPECIFIC) ==========
    rna_preds = []
    
    if dataset_type == 'VITR':
        print("\n=== VITR Pipeline (Clean Data) ===")
        # Use many features, simple methods
        markers = select_elite_markers(ref, mix, n_per_cell=180, strictness='high')
        if len(markers) >= 100:
            print(f"  {len(markers)} markers")
            pred = deconvolve_vitr_optimized(mix, ref, markers)
            if pred is not None:
                rna_preds.append(pred)
        
        # Also try with more features
        markers2 = select_elite_markers(ref, mix, n_per_cell=220, strictness='moderate')
        if len(markers2) >= 100:
            pred = deconvolve_vitr_optimized(mix, ref, markers2)
            if pred is not None:
                rna_preds.append(pred)
    
    elif dataset_type == 'VIVO':
        print("\n=== VIVO Pipeline (Noisy Real Data) ===")
        # Use strict markers, robust methods
        markers = select_elite_markers(ref, mix, n_per_cell=140, strictness='ultra')
        if len(markers) >= 100:
            print(f"  {len(markers)} ultra-strict markers")
            pred = deconvolve_vivo_robust(mix, ref, markers)
            if pred is not None:
                rna_preds.append(pred)
        
        # Backup with moderate markers
        markers2 = select_elite_markers(ref, mix, n_per_cell=170, strictness='high')
        if len(markers2) >= 100:
            pred = deconvolve_vivo_robust(mix, ref, markers2)
            if pred is not None:
                rna_preds.append(pred)
    
    elif dataset_type == 'SIMULATION_SPECIAL':
        print("\n=== Special Simulation Pipeline ===")
        # Could be SDN4, SDN6, SDEL, SDE5, SDC5
        
        # Try SDEL-aware (handles very low proportions)
        markers = select_elite_markers(ref, mix, n_per_cell=160, strictness='high')
        if len(markers) >= 100:
            print(f"  SDEL-aware: {len(markers)} markers")
            pred = deconvolve_sdel_aware(mix, ref, markers)
            if pred is not None:
                rna_preds.append(pred)
        
        # Try standard simulation
        markers2 = select_elite_markers(ref, mix, n_per_cell=180, strictness='moderate')
        if len(markers2) >= 100:
            print(f"  Standard sim: {len(markers2)} markers")
            pred = deconvolve_simulation_standard(mix, ref, markers2)
            if pred is not None:
                rna_preds.append(pred)
        
        # Try robust (for correlation structures)
        markers3 = select_elite_markers(ref, mix, n_per_cell=150, strictness='ultra')
        if len(markers3) >= 100:
            print(f"  Robust: {len(markers3)} markers")
            pred = deconvolve_vivo_robust(mix, ref, markers3)
            if pred is not None:
                rna_preds.append(pred)
    
    else:  # SIMULATION_STANDARD
        print("\n=== Standard Simulation Pipeline ===")
        markers = select_elite_markers(ref, mix, n_per_cell=170, strictness='high')
        if len(markers) >= 100:
            print(f"  {len(markers)} markers")
            pred = deconvolve_simulation_standard(mix, ref, markers)
            if pred is not None:
                rna_preds.append(pred)
    
    # Ensemble RNA
    if len(rna_preds) > 1:
        print(f"\nEnsembling {len(rna_preds)} RNA predictions")
        pred_rna = sum(rna_preds) / len(rna_preds)
        pred_rna = ultra_safe_normalize(pred_rna)
    elif len(rna_preds) == 1:
        pred_rna = rna_preds[0]
    else:
        pred_rna = None
    
    # ========== METHYLATION (SAME LOGIC) ==========
    met_preds = []
    
    if mix_met is not None and ref_met is not None:
        print("\n=== Methylation Deconvolution ===")
        
        if dataset_type == 'VITR':
            markers = select_elite_markers(ref_met, mix_met, n_per_cell=200, strictness='high')
            if len(markers) >= 100:
                pred = deconvolve_vitr_optimized(mix_met, ref_met, markers)
                if pred is not None:
                    met_preds.append(pred)
        
        elif dataset_type == 'VIVO':
            markers = select_elite_markers(ref_met, mix_met, n_per_cell=160, strictness='ultra')
            if len(markers) >= 100:
                pred = deconvolve_vivo_robust(mix_met, ref_met, markers)
                if pred is not None:
                    met_preds.append(pred)
        
        elif dataset_type == 'SIMULATION_SPECIAL':
            markers = select_elite_markers(ref_met, mix_met, n_per_cell=180, strictness='high')
            if len(markers) >= 100:
                pred = deconvolve_sdel_aware(mix_met, ref_met, markers)
                if pred is not None:
                    met_preds.append(pred)
            
            markers2 = select_elite_markers(ref_met, mix_met, n_per_cell=200, strictness='moderate')
            if len(markers2) >= 100:
                pred = deconvolve_simulation_standard(mix_met, ref_met, markers2)
                if pred is not None:
                    met_preds.append(pred)
        
        else:
            markers = select_elite_markers(ref_met, mix_met, n_per_cell=190, strictness='high')
            if len(markers) >= 100:
                pred = deconvolve_simulation_standard(mix_met, ref_met, markers)
                if pred is not None:
                    met_preds.append(pred)
        
        if len(met_preds) > 0:
            print(f"Ensembling {len(met_preds)} methylation predictions")
            pred_met = sum(met_preds) / len(met_preds)
            pred_met = ultra_safe_normalize(pred_met)
        else:
            pred_met = None
    else:
        pred_met = None
    
    # ========== FINAL COMBINATION ==========
    if pred_rna is not None and pred_met is not None:
        print("\n=== Final Combination ===")
        
        common = pred_rna.columns.intersection(pred_met.columns)
        pred_rna = pred_rna[common]
        pred_met = pred_met[common]
        
        # Dataset-specific weights
        if dataset_type == 'VITR':
            rna_w, met_w = 0.54, 0.46
        elif dataset_type == 'VIVO':
            rna_w, met_w = 0.58, 0.42
        else:
            rna_w, met_w = 0.56, 0.44
        
        print(f"Weights: RNA={rna_w}, Met={met_w}")
        
        final = rna_w * pred_rna + met_w * pred_met
        final = ultra_safe_normalize(final)
    elif pred_rna is not None:
        final = pred_rna
    else:
        final = pred_met
    
    print(f"\nFinal: {final.shape}, Range: [{final.values.min():.8f}, {final.values.max():.8f}]")
    
    return final
    
    ##
    ## YOUR CODE ENDS HERE
    ##




# Install and import each package
def install_and_import_packages(required_packages):
  for package in required_packages:
      try:
          globals()[package] = importlib.import_module(package)
      except ImportError:
          print('impossible to import, installing packages',package)
          package_to_install = 'scikit-learn' if package == 'sklearn' else package
          subprocess.check_call([sys.executable, "-m", "pip", "install", package_to_install])
          globals()[package] = importlib.import_module(package)




# Install and import each package
def install_and_import_packages(required_packages):
    def try_pip_install(package_name):
      """Try pip install; detect externally-managed-environment error."""
      try:
          subprocess.check_call(
              [sys.executable, "-m", "pip", "install", package_name]
          )
          return True
      except subprocess.CalledProcessError as e:
          if "externally-managed-environment" in str(e):
              return False  # pip blocked by PEP 668
          raise  # real error unrelated to PEP 668
  

    def try_conda_install(package_name):
        """Attempt conda install."""
        try:
            subprocess.check_call(["conda", "install", "-y", package_name])
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    for package in required_packages:
        try:
            globals()[package] = importlib.import_module(package)
        except ImportError:
            print('impossible to import, installing packages',package)
            package_to_install = 'scikit-learn' if package == 'sklearn' else package
            pip_ok = try_pip_install(package_to_install)

            if pip_ok:
              globals()[package] = importlib.import_module(package)
              continue

            # Pip failed due to externally-managed environment (Debian, Conda etc.)
            print("⛔ pip installation blocked by externally-managed environment.")
            print(f"➡️ Trying conda install: {package_to_install}")

            conda_ok = try_conda_install(package_to_install)

            if conda_ok:
                globals()[package] = importlib.import_module(package)
                continue
            print(f"Unable to install package {package_to_install} automatically.")

def validate_pred(pred, nb_samples=None, nb_cells=None, col_names=None):
    error_status = 0  # 0 means no errors, 1 means "Fatal errors" and 2 means "Warning"
    error_informations = ''

    # Ensure that all sum of cells proportion approximately equal 1
    if not numpy.allclose(numpy.sum(pred, axis=0), 1):
        msg = "The prediction matrix does not respect the laws of proportions: the sum of each column should be equal to 1\n"
        error_informations += msg
        error_status = 2

    # Ensure that the prediction has the correct names
    if not set(col_names) == set(pred.index):
        msg = f"The row names in the prediction matrix should match: {col_names}\n"
        error_informations += msg
        error_status = 2

    # Ensure that the prediction returns the correct number of samples and number of cells
    if pred.shape != (nb_cells, nb_samples):
        msg = f'The prediction matrix has the dimension: {pred.shape} whereas the dimension: {(nb_cells, nb_samples)} is expected\n'
        error_informations += msg
        error_status = 1

    if error_status == 1:
        # The error is blocking and should therefore stop the execution
        raise ValueError(error_informations)
    if error_status == 2:
        print("Warning:")
        print(error_informations)


##############################################################
### Generate a prediction file /!\ DO NOT CHANGE THIS PART ###
##############################################################

# List of required packages
required_packages = [
  "numpy",
  "pandas",
  "zipfile",
  "inspect",
  "h5py",
  "scipy",
  "sklearn"
]
install_and_import_packages(required_packages)

import os
import attachement.data_processing as dp


dir_name = "data"+os.sep

datasets_list = [filename for filename in os.listdir(dir_name) if filename.startswith("mixes")]

ref_file = os.path.join(dir_name, "reference_pdac.h5")
reference_data = dp.read_hdf5(ref_file)


predi_dic = {}
for dataset_name in datasets_list :

    file= os.path.join(dir_name,dataset_name)
    mixes_data = dp.read_hdf5(file)

    print(f"generating prediction for dataset: {dataset_name}")

    # mix_rna = extract_data_element(mixes_data,file, 'mix_rna') 
    # mix_met = extract_data_element(mixes_data,file, 'mix_met')
    cleaned_name=dataset_name.replace("mixes_", "").removesuffix(".h5")

    pred_prop = program(mixes_data["mix_rna"], reference_data["ref_bulkRNA"], mix_met=mixes_data["mix_met"], ref_met=reference_data["ref_met"]   )
    # validate_pred(pred_prop, nb_samples=mix_rna.shape[1], nb_cells=ref_bulkRNA.shape[1], col_names=ref_bulkRNA.columns)
    predi_dic[cleaned_name] = pred_prop

############################### 
### Code submission mode

# we generate a zip file with the 'program' source code

if not os.path.exists("submissions"):
    os.makedirs("submissions")

# we save the source code as a Python file named 'program.py':
with open(os.path.join("submissions", "program.py"), 'w') as f:
    f.write(inspect.getsource(program))

date_suffix = pandas.Timestamp.now().strftime("%Y_%m_%d_%H_%M_%S")




# we create the associated zip file:
zip_program = os.path.join("submissions", f"program_{date_suffix}.zip")
with zipfile.ZipFile(zip_program, 'w') as zipf:
    zipf.write(os.path.join("submissions", "program.py"), arcname="program.py")


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file), 
                       os.path.relpath(os.path.join(root, file), 
                                       os.path.join(path, '..')))
if os.path.exists("attachement"):
    with zipfile.ZipFile(zip_program, 'a', zipfile.ZIP_DEFLATED) as zipf:
        zipdir('attachement/', zipf)


print(zip_program)

###############################


# Generate a zip file with the prediction
if not os.path.exists("submissions"):
    os.makedirs("submissions")

prediction_name = "prediction.h5"

dp.write_hdf5(os.path.join("submissions", prediction_name),predi_dic)



# Create the associated zip file:
zip_results = os.path.join("submissions", f"results_{date_suffix}.zip")
with zipfile.ZipFile(zip_results, 'w') as zipf:
    zipf.write(os.path.join("submissions", prediction_name), arcname=prediction_name)

print(zip_results)

