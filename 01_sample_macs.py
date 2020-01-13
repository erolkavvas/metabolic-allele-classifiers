"""
01_sample_macs.py: code for generating samples of Metabolic Allele Classifiers (MACs).
The code has been generated in such a way such that the total set of samples can be built with additive runs of this program.
To execute the code:

python 01_sample_macs.py -f INPUT_DIR -s NUM_SAMPLES -o OUT_DIR [-a ACTION_NUM --nabound --fracopt --fracpfba --openexch --fvaconstraints]

e.g., tightdude$ python 01_sample_macs.py -f input_data -s 2 -o mnc_ensemble_0 --testsize 0.9

- INPUT_DIR: Path to folder containing the following files named exactly below
    cobra model                 (filename: 'MODEL_FILE.json')       REQUIRED
    strain allele matrix        (filename: 'X_ALLELES_FILE.csv')    REQUIRED
    strain phenotypes matrix    (filename: 'Y_PHENOTYPES_FILE.csv') REQUIRED
    GENE_LIST_FILE              (filename: 'GENE_LIST_FILE.csv')    OPTIONAL - highly recommended <200 genes. otherwise sample deeply

- OUT_DIR: Path to MNC ensemble directory. Parameters and output directory name must be consistent if running gen_mnc_samples multiple times for same ensemble
- ACTION_NUM: Number of total upper and lower bound constraints per allele. Must be an even number!
- NUM_SAMPLES: Number of samples to generate (recommend 2 for first run)
"""
print("... running 01_sample_macs.py ... loading packages and input data ...")
import cobrascape.species as cs
from cobra.io import load_json_model
import pandas as pd
import sys,os,argparse,resource,warnings
from sklearn.model_selection import train_test_split
### MNC objective estimation packages
from tqdm import tqdm
from cobrascape.species import load_json_obj, save_json_obj
from os.path import isfile, join
from os import listdir
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.discrete.discrete_model import Logit #.fit
from statsmodels.tools import add_constant
### other shit 
resource.setrlimit(resource.RLIMIT_NOFILE, (10000,-1))
warnings.filterwarnings("ignore")  # sklearn gives hella warnings.

# Argument parsing
parser = argparse.ArgumentParser(description='Generate MAC samples')
### Required parameters
parser.add_argument('-f',dest='input_dir',required=True,
                    help='Path to folder containing MAC data inputs: cobra model (MODEL_FILE), strain allele matrix (X_ALLELES_FILE), strain phenotypes matrix (Y_PHENOTYPES_FILE), and optionally GENE_LIST')
parser.add_argument('-s',type=int,dest='num_samples',required=True,
                    help='Number of samples to generate (recommend 2 for test run)')
parser.add_argument('-o',dest='out_dir',required=True,
                    help='Output file directory for the MNC ensemble. If it already exists, parameters inputs will be checked to see if everything is the same (default: current directory)')

### Optional parameters (see Methods section of Kavvas et al 2020 for parameters utilized in study)
parser.add_argument('-a',type=int,dest='action_num',required=False, default=4,
                    help='Total number of upper and lower bound constraints per allele. Must be an even number! (default: 4')
parser.add_argument('--testsize',type=float,dest='testsize', default=0.66,
                    help='test_size parameter in sklearn train_test_split function (default: 0.66)')
parser.add_argument('--nabound', dest='add_na_bound', action='store_true',
                    help='Determines whether each allele can map to a no_change constraint (default: False). Type --nabound to set to True')
parser.add_argument('--fracopt',type=float,dest='FVA_frac_opt', default=0.0,
                    help='fraction_of_optimum parameter in cobrapy flux_variability_analysis function (default: 0.0)')
parser.add_argument('--fracpfba',dest='FVA_pfba_fract', default=1.1,
                    help='pfba_factor parameter in cobrapy flux_variability_analysis function (default: 1.1)')
parser.add_argument('--openexch',type=bool,dest='FVA_open_exchanges', default=True,
                    help='open_exchanges parameter in cobrapy find_blocked_reactions function (default: True)')
parser.add_argument('--fvaconstraints',type=bool,dest='FVA_constriants', default=True,
                    help='Decides whether or not to apply the FVA constraints above to the base cobra model (default: True)')
### Optional parameters below are for the objective estimation step once the popFVA samples are generated for each allele-constraint map
parser.add_argument('--popfvascale', dest='popFVA_STANDARDIZE', action='store_true',
                    help='Decides how the sampled popFVA landscape will be scaled prior to logistic regression. Type --popfvascale to specify True. False corresponds to minmax scaling while True corresponds to standard z-score scaling (default: False)')
parser.add_argument('--l1alpha',type=float,dest='l1alpha', default=0.5,
                    help='L1-regularization penalty for statsmodels regularized logit. Larger alpha values correspond to harsher L1-penalization and more sparse solutions (default: 0.5)')
parser.add_argument('--pcancomps',type=float,dest='pcancomps', default=0.9,
                    help='Minimum percent of total variations decided by PCA decomposition of popFVA landscape. n_components parameter for sklearn PCA function (default: 0.9)')
parser.add_argument('--gensamples', dest='gensamples', action='store_false',
                    help='Whether to build species object and sample landscapes. Type --gensamples to specify False. Useful if samples already generated and you want to go straight to computing MNC objectives (default: True)')
parser.add_argument('--estobjective', dest='estobjective', action='store_false',
                    help='Whether to estimate MAC objectives for generated samples. Type --estobjective to specify False. Useful if you only want to generate samples without approximating objective at end (default: True)')
### Should I add arguments for generating samples and estimating objectives?
args = parser.parse_args()

ENSEMBLE_DIR = args.out_dir
if not os.path.exists(ENSEMBLE_DIR+"/"):
    print('\t... creating ensemble directory:'+ENSEMBLE_DIR+"/")
    os.makedirs(ENSEMBLE_DIR+"/")
    print('\t... saving parameters to ensemble directory')
    with open(ENSEMBLE_DIR+'/mnc_ensemble_args.txt', 'w') as f:
        f.write('\n'.join(sys.argv[1:]))
    ### save to json
    args_dict = {
        "action_num": args.action_num, 
        'nabound': args.add_na_bound,
        'popFVA_STANDARDIZE': args.popFVA_STANDARDIZE,
        'testsize': args.testsize
    }
    save_json_obj(args_dict, ENSEMBLE_DIR+"/mnc_ensemble_args.json")
else:
    exit_script=False
    args_dict = load_json_obj(ENSEMBLE_DIR+"/mnc_ensemble_args.json")

    if str(args_dict["nabound"])!=str(args.add_na_bound):
        print("--nabound argument is different!")
        exit_script=True

    if args_dict["action_num"]!=args.action_num:
        print("--action_num argument is different!")
        exit_script=True

    if args_dict["popFVA_STANDARDIZE"]!=args.popFVA_STANDARDIZE:
        print("--popfvascale argument is different!", args_dict["popFVA_STANDARDIZE"], str(args.popFVA_STANDARDIZE))
        exit_script=True

    if args_dict["testsize"]!=args.testsize:
        print("--testsize argument is different!")
        exit_script=True

    ### Exist script if any of the if statements above are entered.
    if exit_script==True:
        print("...EXITING... change input parameters to match MNC ensemble described in output folder")
        sys.exit()

DATA_DIR = args.input_dir + "/"
MODEL_FILE = DATA_DIR + 'MODEL_FILE.json'
X_ALLELES_FILE = DATA_DIR + 'X_ALLELES_FILE.csv'            # "iEK1011_drugTesting_media.json"
Y_PHENOTYPES_FILE = DATA_DIR + 'Y_PHENOTYPES_FILE.csv'      # 
GENE_LIST_FILE = DATA_DIR + "GENE_LIST_FILE.csv" # If not found, will use ALL genes in cobra model
MODEL_SAMPLES_FILENAME = "base_flux_samples.csv" # If not found in ENSEMBLE_DIR, script will perform flux sampling.

action_num = args.action_num # 4 
ADD_NA_BOUND = args.add_na_bound  # False
NUM_SAMPLES = args.num_samples # 10

FVA_frac_opt = args.FVA_frac_opt # 0.1 # Decides the minimum flux required through biomass production
FVA_pfba_fract = args.FVA_pfba_fract # 1.5 # Decides allowable flux space based on upper bounding the total sum of fluxes
OPEN_EXCHANGE_FVA = args.FVA_open_exchanges # True. Whether to open exchange fluxes for FVA calculations. 
FVA_CONSTRAINTS = args.FVA_constriants # True. Whether to apply the following constraints above.
FVA_CONSTRAINTS = args.FVA_constriants
popFVA_STANDARDIZE = args.popFVA_STANDARDIZE
logreg_l1alpha = args.l1alpha
pca_n_components = args.pcancomps
GENERATE_SAMPLES = args.gensamples
ESTIMATE_MNC_OBJECTIVE = args.estobjective

X_species = pd.read_csv(X_ALLELES_FILE, index_col = 0)
Y_phenotypes = pd.read_csv(Y_PHENOTYPES_FILE, index_col = 0)

# X_df = X_species.copy()
# Y_df = Y_phenotypes.reindex(X_df.index.tolist()).copy()
X_train, X_test, y_train, y_test = train_test_split(X_species, Y_phenotypes.reindex(X_species.index.tolist()), test_size=args.testsize, random_state=42)
if not os.path.exists(ENSEMBLE_DIR+"/X_train.csv"):
    X_train.to_csv(ENSEMBLE_DIR+"/X_train.csv")
    X_test.to_csv(ENSEMBLE_DIR+"/X_test.csv")
    y_train.to_csv(ENSEMBLE_DIR+"/y_train.csv")
    y_test.to_csv(ENSEMBLE_DIR+"/y_test.csv")

X_species_final = X_train
Y_pheno_final = y_train
print("input: (G)enetic variant matrix= (strains: %d, alleles: %d)" % (X_species_final.shape[0], X_species_final.shape[1]))
print("input: Class distribution for each phenotype")
for pheno in Y_pheno_final.columns:
    print("\t",pheno, "train:", (y_train[pheno].value_counts().to_dict()), 
          "test:", (y_test[pheno].value_counts().to_dict()))

COBRA_MODEL = load_json_model(MODEL_FILE)
print("input: (S)toichimetric genome-scale model= (genes: %d, reactions: %d, metabolites: %d)" % (len(COBRA_MODEL.genes), 
    len(COBRA_MODEL.reactions), len(COBRA_MODEL.metabolites)))

### The desired media condition should already be initialized.
print("COBRA_MODEL.medium: ", COBRA_MODEL.medium)

sol = COBRA_MODEL.optimize()
print("\t... before cleaning (objective_value: %f)" % (sol.objective_value))
### Clean base model and apply FVA constriants
COBRA_MODEL = cs.clean_base_model(COBRA_MODEL, open_exchange=OPEN_EXCHANGE_FVA, verbose=False)
sol = COBRA_MODEL.optimize()
print("\t... after cleaning (objective_value: %f)" % (sol.objective_value))
if FVA_CONSTRAINTS==True:
    COBRA_MODEL, fva_df = cs.init_fva_constraints(COBRA_MODEL,opt_frac=FVA_frac_opt, pfba_fact=FVA_pfba_fract, verbose=False)
    sol = COBRA_MODEL.optimize()
    print("\t... after fva constraints (objective_value: %f)" % (sol.objective_value))
    print("\t... filtered GEM= (genes: %d, reactions: %d, metabolites: %d)" % (len(COBRA_MODEL.genes), 
        len(COBRA_MODEL.reactions), len(COBRA_MODEL.metabolites)))


# ENSEMBLE_DIR = "ens_strains"+str(len(SPECIES_MODEL.strains))+"_alleles"+str(len(players))+"_actions"+str(action_num)
POPFVA_SAMPLES_DIR = ENSEMBLE_DIR+"/popfva_samples/"
print("output dir: %s" % (ENSEMBLE_DIR))
if not os.path.exists(POPFVA_SAMPLES_DIR):
    print('\t... creating sampling directory:'+POPFVA_SAMPLES_DIR)
    os.makedirs(POPFVA_SAMPLES_DIR)


pheno_list = Y_pheno_final.columns
ALLELE_PHENO_FILE = ENSEMBLE_DIR+"/allele_pheno_data/"

### ------------------------------------------------------------
### Sample allele-constraint maps and popFVA landscapes for MNCs
### ------------------------------------------------------------
if GENERATE_SAMPLES==True:

    MODEL_SAMPLES_FILE = ENSEMBLE_DIR+"/"+MODEL_SAMPLES_FILENAME
    if not os.path.exists(MODEL_SAMPLES_FILE):
        from cobra import sampling
        print("\t... generating flux samples for base cobra model...(may take >10 minutes). Only performed once!")
        rxn_flux_samples_ARCH = sampling.sample(COBRA_MODEL, 1000,method='achr', 
                                          thinning=100, processes=6, seed=None)
        print("\t... saving flux samples for base cobra model: ", MODEL_SAMPLES_FILE)
        rxn_flux_samples_ARCH.to_csv(MODEL_SAMPLES_FILE)

    ENSEMBLE_BASEMODEL_FILE = ENSEMBLE_DIR+"/base_cobra_model.json"
    if not os.path.exists(ENSEMBLE_BASEMODEL_FILE):
        from cobra.io import save_json_model
        print("\t... saving base cobra model: ", ENSEMBLE_BASEMODEL_FILE)
        save_json_model(COBRA_MODEL,ENSEMBLE_BASEMODEL_FILE)
        
    base_flux_samples = pd.read_csv(MODEL_SAMPLES_FILE,index_col=0)

    ### Create Species object
    SPECIES_MODEL = cs.Species("species_obj")
    COBRA_MODEL.solver = "glpk"
    SPECIES_MODEL.base_cobra_model = COBRA_MODEL
    SPECIES_MODEL.load_from_matrix(X_species_final, filter_model_genes=True, allele_gene_sep="_")
    for allele in SPECIES_MODEL.alleles:
        allele.cobra_gene = allele.id.split("_")[0]
        
    ### Determines the genes that the MNC will model the alleles of
    if not os.path.exists(GENE_LIST_FILE):
        print('\t... no gene list given, using all genes!'+POPFVA_SAMPLES_DIR)
    else:
        amr_gene_df = pd.read_csv(GENE_LIST_FILE,index_col=0)
        amr_gene_list = amr_gene_df.index.tolist()
        print("# of genes:",len(amr_gene_list))
        gene_list = amr_gene_list # e.g., ["Rv1908c", "Rv2245", "Rv1483"]
        amr_gene_df.to_csv(ENSEMBLE_DIR+"/GENE_LIST_FILE.csv")

    players, player_reacts, player_metabs = cs.get_gene_players(gene_list, SPECIES_MODEL, verbose=True)
    ### Update the strains... takes long time... ensure that 
    SPECIES_MODEL.update_strains_cobra_model()

    ### Save genetic variant matrix and strain phenotypes for each case.
    allele_list = [x.id for x in players]

    ## ALLELE_PHENO_FILE = ENSEMBLE_DIR+"/allele_pheno_data/"
    if not os.path.exists(ALLELE_PHENO_FILE):
        print('\t... creating sampling directory:'+ALLELE_PHENO_FILE)
        os.makedirs(ALLELE_PHENO_FILE)
    for pheno_id in pheno_list:
        G_VARIANT_MATRIX_FILE = ALLELE_PHENO_FILE+"/allele_df_"+pheno_id+".csv"
        PHENO_MATRIX_FILE = ALLELE_PHENO_FILE+"/pheno_df_"+pheno_id+".csv"
        X_filtered, Y_filtered = cs.filter_pheno_nan(X_species_final, Y_pheno_final, pheno_id)
        if not os.path.exists(G_VARIANT_MATRIX_FILE):
            X_filtered.loc[:,allele_list].to_csv(G_VARIANT_MATRIX_FILE)
        if not os.path.exists(PHENO_MATRIX_FILE):
            pd.DataFrame(Y_filtered).to_csv(PHENO_MATRIX_FILE) # , header=True

    ### --- Generate ensemble of random allele-constraint maps and their corresponding popFVA landscapes
    pool_obj = cs.sample_species(SPECIES_MODEL, POPFVA_SAMPLES_DIR, players, base_flux_samples,
                                 fva_rxn_set="var_reacts", start_samp=None, samples_n=NUM_SAMPLES, fva=True,
                                 fva_frac_opt=FVA_frac_opt, action_n=action_num, add_na_bound=ADD_NA_BOUND)

    print("...sampling of MNC allele-constraint maps and popFVA landscapes finished! ...")
else:
    print("...--gensamples set to False...... skipping sampling step")

### ------------------------------------------------------------
### Estimate objective for each sampled MNC in folder.
### ------------------------------------------------------------
if ESTIMATE_MNC_OBJECTIVE==True:

    ### Load in the genetic variant matrix and AMR phenotypes for each case.
    pheno_to_data2d_dict = {}
    pheno_to_Y_dict = {}
    for pheno_id in pheno_list:
        G_VARIANT_MATRIX_FILE = ALLELE_PHENO_FILE+"/allele_df_"+pheno_id+".csv"
        PHENO_MATRIX_FILE = ALLELE_PHENO_FILE+"/pheno_df_"+pheno_id+".csv"
        pheno_to_data2d_dict.update({pheno_id: pd.read_csv(G_VARIANT_MATRIX_FILE,index_col=0)})
        pheno_to_Y_dict.update({pheno_id: pd.read_csv(PHENO_MATRIX_FILE,index_col=0)[pheno_id]}) ### series

    ### Save list of samples already assessed, so we can skip these and build a matrix for only the new ones.
    onlyfiles = [f for f in listdir(POPFVA_SAMPLES_DIR) if isfile(join(POPFVA_SAMPLES_DIR, f))]
    onlyfiles = [f for f in onlyfiles if f != ".DS_Store"]
    int_list = [int(x.split("_")[1]) for x in onlyfiles if len(x.split("_"))>2]
    if len(int_list)>0:
        total_samples = max(int_list)+1
    else:
        total_samples = 0
        print("\t... nothing in %s" % (POPFVA_SAMPLES_DIR))

    ENSEMBLE_MAP_ASSESS = ENSEMBLE_DIR+"/popfva_assessment/"
    MNC_OBJ_DIR = ENSEMBLE_DIR+"/mnc_objectives/"
    for ens_folder in [ENSEMBLE_MAP_ASSESS, MNC_OBJ_DIR]:
        if not os.path.exists(ens_folder):
            print('\t... creating %s' % (ens_folder))
            os.makedirs(ens_folder)
        else:
            print("dir assess: %s" % (ens_folder))

    ### Run loop... should parallelize
    ENSEMBLE_DATA_DICT = {}
    for landscape_sample_num in tqdm(range(total_samples)[:]): #
        sample_id = "sampled_map_"+str(landscape_sample_num)
        ENSEMBLE_DATA_DICT[sample_id] = {}
        
        if not os.path.exists(ENSEMBLE_MAP_ASSESS+"sample_"+str(landscape_sample_num)+"_map_assess.json"):
            variant_dec_file = POPFVA_SAMPLES_DIR+"sample_"+str(landscape_sample_num)+"_varDecision.json"

            if not os.path.exists(variant_dec_file):
                print('file "%s" does not exist' % (variant_dec_file))
            else:
                variant_dec_dict = load_json_obj(variant_dec_file)
                ENSEMBLE_DATA_DICT[sample_id].update(variant_dec_dict)

                fva_landscape_file = POPFVA_SAMPLES_DIR+"sample_"+str(landscape_sample_num)+"_FVA.json"
                fva_landscape_dict = load_json_obj(fva_landscape_file)
                ENSEMBLE_DATA_DICT[sample_id].update({"fva_landscape_file": fva_landscape_file})

                obj_val_list = {}
                for strain_id, strain_fva_dict in fva_landscape_dict.items(): # this has items b/c its already a dict
                    obj_val_list[strain_id] = {}
                    for rxn, max_min_dict in strain_fva_dict.items():
                        obj_val_list[strain_id].update({rxn+"_max":float(format(max_min_dict["maximum"],'.10f')), 
                                                        rxn+"_min":float(format(max_min_dict["minimum"],'.10f'))})
                popfva_df = pd.DataFrame.from_dict(obj_val_list, orient="index")

                ### Loop between each map
                for pheno_id, genetic_variant_df in pheno_to_data2d_dict.items():
                    genetic_variant_df = pheno_to_data2d_dict[pheno_id]
                    Y_pheno_test = pheno_to_Y_dict[pheno_id]
                    Y_pheno_test_reindexed = Y_pheno_test.reindex(genetic_variant_df.index)
                    popfva_df_reindexed = popfva_df.reindex(genetic_variant_df.index)

                    """ Type of scaling to occur before fitting regularized logistic regression
                        see link below for discussion on which type of scaling to use:
                        https://stats.stackexchange.com/questions/69157/why-do-we-need-to-normalize-data-before-principal-component-analysis-pca
                    """
                    if popFVA_STANDARDIZE==True:
                        landscape_scaler = StandardScaler() # Standardization Z-score
                    else:
                        landscape_scaler = MinMaxScaler() # Normalization 0-1 scaling

                    popfva_df_reind_scaled = landscape_scaler.fit_transform(popfva_df_reindexed)
                    popfva_df_reind = pd.DataFrame(popfva_df_reind_scaled, index=popfva_df_reindexed.index, columns=popfva_df_reindexed.columns)
                    X_standardscaled_SAMPLE = popfva_df_reind.reindex(genetic_variant_df.index)
                    
                    # set Y and X variable
                    y = Y_pheno_test_reindexed.astype(int)
                    y.dropna(axis=0, how='all', inplace=True)
                    X = X_standardscaled_SAMPLE
                    X = X.reindex(y.index)
                    pca = PCA(n_components=pca_n_components, svd_solver = 'full', whiten=True) ## Primary purpose of PCA: to remove collinearity in popFVA landscapes
                    X_pca = pca.fit_transform(X)
                    PCA_weight_dict = pd.DataFrame(pca.components_,index=range(len(pca.components_)),columns=X.columns).T.to_dict()
                    """ Perform regularized logistic regression model:
                        NOTE from Hastie, Tibshirani, Friedman (The ridge solutions are not equivariant under scaling of the inputs, 
                        and so one normally standardizes the inputs before solving.)
                    """
                    lgit = Logit(y, add_constant(X_pca)).fit_regularized(maxiter=1500, disp=False, alpha=logreg_l1alpha)

                    ENSEMBLE_DATA_DICT[sample_id].update({
                        "AIC_"+pheno_id: lgit.aic,
                        "BIC_"+pheno_id: lgit.bic,
                        "prsquared_"+pheno_id: lgit.prsquared,
                        "loglikelihood_"+pheno_id: lgit.llr,
                        "LLR_pval_"+pheno_id: lgit.llr_pvalue,
                        "p_values_"+pheno_id: str(lgit.pvalues.to_dict()),
                        "coefs_"+pheno_id: str(lgit.params.to_dict()),
                        "std_err_"+pheno_id: str(lgit.bse.to_dict()),
                        "PCA_comp_dict_"+pheno_id: str(PCA_weight_dict)
                    })

                    ### Approximate MNC objective using the Logistic regression popFVA PCA coefficients
                    comp_dict = {"x"+str(k+1):v for k, v in PCA_weight_dict.items() }
                    pca_sample_df = pd.DataFrame.from_dict(comp_dict)
                    coef_vector = pd.DataFrame.from_dict(lgit.params.to_dict(),orient="index")[0]
                    coef_vector.drop(["const"],inplace=True)
                    col_id = sample_id
                    # r_vector = np.dot(pca_sample_df.T, coef_vector)
                    r_df = pd.DataFrame(np.dot(pca_sample_df, coef_vector), index=pca_sample_df.index, columns=[col_id])
                    r_df.sort_values([col_id], inplace=True)
                    r_df.dropna(inplace=True)
                    obj_dict_fn = MNC_OBJ_DIR+"obj_"+sample_id+"__"+pheno_id+".json"
                    save_json_obj(r_df.to_dict(), obj_dict_fn)
                    
                ### Save the Logistic regression fit for the sameple to ENSEMBLE_MAP_ASSESS
                save_json_obj(ENSEMBLE_DATA_DICT[sample_id], ENSEMBLE_MAP_ASSESS+"sample_"+str(landscape_sample_num)+"_map_assess.json")
                ENSEMBLE_DATA_DICT[sample_id] = {}

    print("...estimation of MNC objectives finished! ...")
else:
    print("...--estobjective set to False... skipping MNC objective approximation")
