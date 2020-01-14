"""
02_eval_macs.py: code for evaluating sampled MACs on either the test set or training set. 

The code has been generated in such a way such that the total set of samples can be built with additive runs of this program.
To execute the code:

python 02_eval_macs.py.py -f MAC_ENSEMBLE_DIR [--testset --bicthresh]

e.g., tightdude$ python 02_eval_macs.py -f mnc_ensemble_0

- MAC_ENSEMBLE_DIR: Path to folder containing an output directory of 01_sample_macs.py
"""
import cobrascape.species as cs
import cobrascape.ensemble as ens
from cobra.io import load_json_model
from cobrascape.species import load_json_obj, save_json_obj
import pandas as pd
import numpy as np
import sys,os,argparse,resource,warnings,itertools
from os import listdir
# import seaborn as sns
# import matplotlib.pyplot as plt
from random import shuffle
from sklearn.metrics import roc_curve, auc
from collections import Counter
from tqdm import tqdm
warnings.filterwarnings("ignore")  # sklearn gives hella warnings.
resource.setrlimit(resource.RLIMIT_NOFILE, (10000,-1))

# Argument parsing
parser = argparse.ArgumentParser(description='Evaluate MAC samples')
### Required parameters
parser.add_argument('-f',dest='mnc_dir',required=True,
                    help='Path to folder containing MAC samples. Should be same as -o argument of 01_sample_macs.py')
### Optional parameters (see Methods section of Kavvas et al 2020 for parameters utilized in study)
parser.add_argument('--testset', dest='train_test', action='store_true',
                    help='Whether to evaluate for training (False) or test (True) set. Type --testset to specify False (default: False)')
parser.add_argument('--bicthresh',type=int,dest='bic_threshold', default=10,
                    help='Delta BIC determines the set of high quality models to perform analysis on. (default: 10). See Burnham and Anderson 2002 book on model selection for further information')

### load args
args = parser.parse_args()

ENSEMBLE_DIR = args.mnc_dir
TESTSET = args.train_test
BIC_cutoff = args.bic_threshold

ensemble_args_dict = load_json_obj(ENSEMBLE_DIR+"/mnc_ensemble_args.json")
action_num = ensemble_args_dict["action_num"] # 4 
ADD_NA_BOUND = ensemble_args_dict["nabound"] # False
STANDARDIZE_ = ensemble_args_dict["popFVA_STANDARDIZE"] # False
print("action_num (%d), nabound (%s), standardize (%s)"%(action_num, str(ADD_NA_BOUND),str(STANDARDIZE_)))

SCALE_POPFVA_=True
pval_threshold= 1.0
load_threshold=0.0
fdr_correction=False
save_data=True

#### IMPORTANT - top_models = pd.read_csv(ENSEMBLE_DIR+"/tables/best_mncs_"+pheno_id+".csv",index_col=0) will fail if best_mncs_ file is not generated before~!
#### write code for getting list of best MNCs for each phenotype
if TESTSET==False:
    print("...loading TRAINING data...")
    X_species_final = pd.read_csv(ENSEMBLE_DIR+"/X_train.csv", index_col = 0)
    Y_pheno_final = pd.read_csv(ENSEMBLE_DIR+"/y_train.csv", index_col = 0)
    file_outtag = "train"
elif TESTSET==True:
    print("...loading TEST data...")
    X_species_final = pd.read_csv(ENSEMBLE_DIR+"/X_test.csv", index_col = 0)
    Y_pheno_final = pd.read_csv(ENSEMBLE_DIR+"/y_test.csv", index_col = 0)
    file_outtag = "test"

print("input: (G)enetic variant matrix= (strains: %d, alleles: %d)" % (X_species_final.shape[0], X_species_final.shape[1]))

COBRA_MODEL = load_json_model(ENSEMBLE_DIR+"/base_cobra_model.json")
print("input: (S)toichimetric genome-scale model= (genes: %d, reactions: %d, metabolites: %d)" % (len(COBRA_MODEL.genes), 
    len(COBRA_MODEL.reactions), len(COBRA_MODEL.metabolites)))

### Create Species object
SPECIES_MODEL = cs.Species("species_obj")
COBRA_MODEL.solver = "glpk"
SPECIES_MODEL.base_cobra_model = COBRA_MODEL
SPECIES_MODEL.load_from_matrix(X_species_final, filter_model_genes=True, allele_gene_sep="_")
for allele in SPECIES_MODEL.alleles:
    allele.cobra_gene = allele.id.split("_")[0]
    
### Get AMR genes of interest
### Determines the genes that the MNC will model the alleles of
GENE_LIST_FILE = ENSEMBLE_DIR+"/GENE_LIST_FILE.csv"
if not os.path.exists(GENE_LIST_FILE):
    print('\t... no gene list given, using all genes!'+POPFVA_SAMPLES_DIR)
else:
    amr_gene_df = pd.read_csv(GENE_LIST_FILE,index_col=0)
    amr_gene_list = amr_gene_df.index.tolist()
    print("# of genes:",len(amr_gene_list))
    gene_list = amr_gene_list # e.g., ["Rv1908c", "Rv2245", "Rv1483"]

players, player_reacts, player_metabs = cs.get_gene_players(gene_list, SPECIES_MODEL, verbose=True)
### Update the strains... takes long time... ensure that 
SPECIES_MODEL.update_strains_cobra_model()

base_flux_samples = pd.read_csv(ENSEMBLE_DIR+"/"+"base_flux_samples.csv",index_col=0)
POPFVA_SAMPLES_DIR = ENSEMBLE_DIR+"/popfva_samples/"
ENSEMBLE_MAP_ASSESS = ENSEMBLE_DIR+"/popfva_assessment/"

### Save genetic variant matrix and AMR phenotypes for each case.
allele_list = [x.id for x in players]
pheno_list = Y_pheno_final.columns

pheno_sim_list = pheno_list ## change this so it can be an input parameter

### Load in the genetic variant matrix and AMR phenotypes for each case.
pheno_to_data2d_dict = {}
pheno_to_Y_dict = {}
if TESTSET==False:
    ALLELE_PHENO_FILE = ENSEMBLE_DIR+"/allele_pheno_data/"
    for pheno_id in pheno_list:
        G_VARIANT_MATRIX_FILE = ALLELE_PHENO_FILE+"/allele_df_"+pheno_id+".csv"
        PHENO_MATRIX_FILE = ALLELE_PHENO_FILE+"/pheno_df_"+pheno_id+".csv"
        pheno_to_data2d_dict.update({pheno_id: pd.read_csv(G_VARIANT_MATRIX_FILE,index_col=0)})
        pheno_to_Y_dict.update({pheno_id: pd.read_csv(PHENO_MATRIX_FILE,index_col=0)[pheno_id]}) ### series
elif TESTSET==True:
    for pheno_id in pheno_list:
        X_filtered, Y_filtered = cs.filter_pheno_nan(X_species_final, Y_pheno_final, pheno_id)
        pheno_to_data2d_dict.update({pheno_id: X_filtered.loc[:,allele_list]})
        pheno_to_Y_dict.update({pheno_id: Y_filtered}) 


def set_sample_pheno_popobj(ensemble_dir, species_mod, sample_id, obj_direct, 
                            pheno_id="isoniazid", ic_id="BIC", pval_threshold=1, load_threshold=0):
    PCA_DATA_DIR = ensemble_dir+"/mnc_objectives/"
    sample_id_num = sample_id.split("_")[-1]
    PCA_SAMPLE_FN = PCA_DATA_DIR+"obj_sampled_map_"+sample_id_num+"__"+pheno_id+".json"
    # print(PCA_SAMPLE_FN)
    sample_pheno_obj = ens.load_json_obj(PCA_SAMPLE_FN)
    
    r_df = pd.DataFrame.from_dict(sample_pheno_obj)
    col_id = "r_"+sample_id
    r_df.columns = [col_id]
    r_df.sort_values([col_id], inplace=True)

    ### Filter dataframe according to PCA loading threshold
    r_filt_df = r_df[abs(r_df)>load_threshold]
    r_filt_df.dropna(inplace=True)

    ### Load allele-constraint map and popFVA landscape for the particular sample
    var_sample_file=str(sample_id)+"_varDecision.json"
    fva_sample_file=str(sample_id)+"_FVA.json"
    POPFVA_SAMPLES_DIR = ensemble_dir+"/popfva_samples/"
    sample_ac = ens.get_sample_constraints(POPFVA_SAMPLES_DIR+var_sample_file)
    sample_popFVA = ens.load_landscape_sample(POPFVA_SAMPLES_DIR+fva_sample_file)
    # print("...set PCA objective")
    species_mod = cs.set_linear_popfva_objective(species_mod, r_filt_df, obj_dir=obj_direct)
    return species_mod, sample_ac, r_filt_df


def get_alleles_rxn(x):
    genes_in_rxn=[y.id for y in SPECIES_MODEL.base_cobra_model.reactions.get_by_id(x).genes]
    rxn_g_list=[]
    for g in gene_list:
        if g in genes_in_rxn:
            rxn_g_list.append(g)
    if len(rxn_g_list)==0:
        return None
    else:
        return rxn_g_list
    
    
def convert_popsolfva_2df(popfva_sol):
    fva_landscape_dict = dict(popfva_sol)
    obj_val_list = {}
    for strain_id, strain_fva_dict in fva_landscape_dict.items(): # this has items b/c its already a dict
        obj_val_list[strain_id] = {}
        for rxn, max_min_dict in strain_fva_dict.items():
            obj_val_list[strain_id].update({rxn+"_max":float(format(max_min_dict["maximum"],'.10f')), 
                                            rxn+"_min":float(format(max_min_dict["minimum"],'.10f'))})
    fva_landscape_df = pd.DataFrame.from_dict(obj_val_list, orient="index")
    return fva_landscape_df


def get_popfva_opt_longform(popfva_opt_df, pheno_to_data2d_dict, pheno_to_Y_dict, 
                            pheno_id="isoniazid", optdirect="max"):
    """ 
        vdir: 1 and 0 values corresponds to maximized and minimized flux through reactions, respectively.
        optdir: 1 and 0 values corresponds to maximized and minimized population objective, respectively.
    
    """
    rxn_max_cols = [x for x in popfva_opt_df.columns if "_max" in x]
    rxn_min_cols = [x for x in popfva_opt_df.columns if "_min" in x]

    popfva_vmax_df = popfva_opt_df[rxn_max_cols]
    popfva_vmax_df.columns = [x.split("_max")[0] for x in popfva_vmax_df.columns]
    popfva_vmax_df["vdir"] = "vmax"
    popfva_vmin_df = popfva_opt_df[rxn_min_cols]
    popfva_vmin_df.columns = [x.split("_min")[0] for x in popfva_vmin_df.columns]
    popfva_vmin_df["vdir"] = "vmin"

    if optdirect=="max":
        popfva_vmax_df["optdir"] = "objmax"
        popfva_vmin_df["optdir"] = "objmax"
    elif optdirect=="min":
        popfva_vmax_df["optdir"] = "objmin"
        popfva_vmin_df["optdir"] = "objmin"
        
    X_popfva_vmin_df, y = ens.return_filt_matrix(popfva_vmin_df, pheno_to_data2d_dict, pheno_to_Y_dict, pheno_id=pheno_id)
    popfva_vmin_df = pd.concat([X_popfva_vmin_df, y],axis=1)
    X_popfva_vmax_df, y = ens.return_filt_matrix(popfva_vmax_df, pheno_to_data2d_dict, pheno_to_Y_dict, pheno_id=pheno_id)
    popfva_vmax_df = pd.concat([X_popfva_vmax_df, y],axis=1)
    return popfva_vmax_df, popfva_vmin_df


def load_samples_assess_df(ENSEMBLE_MAP_ASSESS, pheno_list):
    ### -------------- LOAD 2 -----------------
    print("...loading SAMPLES_ASSESS_DF to identify minimum BIC or AIC MNCs")
    onlyfiles = [f for f in listdir(ENSEMBLE_MAP_ASSESS) if os.path.isfile(os.path.join(ENSEMBLE_MAP_ASSESS, f))]
    onlyfiles = [f for f in onlyfiles if f != ".DS_Store"]
    samplesAfter = [f for f in onlyfiles if "sample_" in f] 

    wanted_keys = []
    ### Options for what we want in SAMPLES_ASSESS_DF
    for pheno_id in pheno_list:
        wanted_keys.extend(["AIC_"+pheno_id, "BIC_"+pheno_id])

    SAMPLES_ASSESS_DF = {}
    for landscape_sample_name in tqdm(samplesAfter):
        landscape_sample_num = landscape_sample_name.split("_")[1]
        sample_id = "sampled_map_"+str(landscape_sample_num)
        landscape_assess_sample_file = ENSEMBLE_MAP_ASSESS+landscape_sample_name

        if os.path.exists(landscape_assess_sample_file):
            landscape_assess_sample = load_json_obj(landscape_assess_sample_file)
            SAMPLES_ASSESS_DF[sample_id] = {}
            SAMPLES_ASSESS_DF[sample_id].update(dict((k, landscape_assess_sample[k]) for k in wanted_keys if k in landscape_assess_sample))

    # transform to pandas dataframe
    SAMPLES_ASSESS_DF = pd.DataFrame.from_dict(SAMPLES_ASSESS_DF,orient="index")
    print("\t... SAMPLES_ASSESS_DF shape: (samples: %d, assess_cols: %d)" % (SAMPLES_ASSESS_DF.shape[0], SAMPLES_ASSESS_DF.shape[1]))
    return SAMPLES_ASSESS_DF


SAMPLES_ASSESS_DF = load_samples_assess_df(ENSEMBLE_MAP_ASSESS, pheno_list)

if not os.path.exists(ENSEMBLE_DIR+"/tables"):
    print('\t... creating tables directory for saving MNC simulations:'+ENSEMBLE_DIR+"/tables/")
    os.makedirs(ENSEMBLE_DIR+"/tables/")

for pheno_id in pheno_sim_list:
    # top_models = pd.read_csv(ENSEMBLE_DIR+"/tables/best_mncs_"+pheno_id+".csv",index_col=0)
    top_models = SAMPLES_ASSESS_DF[["BIC_"+pheno_id]].copy()
    top_models = top_models-top_models.min()
    top_models = top_models[top_models["BIC_"+pheno_id]<BIC_cutoff]
    sample_sim_list = ["sample_"+x.split("_")[-1] for x in top_models.index.tolist()]
    print(pheno_id, "(# of hq models=%d)"%(len(sample_sim_list)))
    
    for sample_id in sample_sim_list:
        print(pheno_id, sample_id)

        for obj_direction in ["max", "min"]:
            ### file_outtag is either "train" or "test"
            save_file_tag = pheno_id+"_"+sample_id+"_OBJdirect-"+obj_direction+"_"+file_outtag+".csv"

            ### First check to see that the simulations have not been generated
            if not os.path.exists(ENSEMBLE_DIR+"/tables/raw_fluxes_"+save_file_tag):

                SPECIES_MODEL, sample_ac_map, popfva_coef = set_sample_pheno_popobj(
                    ENSEMBLE_DIR, SPECIES_MODEL, sample_id, obj_direction, pheno_id=pheno_id, ic_id="BIC", 
                    pval_threshold=pval_threshold, load_threshold=load_threshold
                )
                print("\t...optimize "+obj_direction)
                popsol = cs.compute_constrained_species(
                    SPECIES_MODEL, sample_ac_map,players,base_flux_samples,fva_rxn_set="all_reacts",
                    fva=False,fva_frac_opt=0.1,action_n=action_num,add_na_bound=ADD_NA_BOUND
                )
                pop_fluxes_df, pop_sprices_df, pop_rcosts_df, pop_sol_df = cs.get_popobj_sol_df(popsol)
                
                if save_data==True:
                    ### Save unscaled fluxes, sprices and reduced costs
                    pop_fluxes_df.to_csv(ENSEMBLE_DIR+"/tables/raw_fluxes_"+save_file_tag)
                    pop_sprices_df.to_csv(ENSEMBLE_DIR+"/tables/raw_sprices_"+save_file_tag)
                    pop_rcosts_df.to_csv(ENSEMBLE_DIR+"/tables/raw_rcosts_"+save_file_tag)            
                
                pop_fluxes_df = ens.scale_df(pop_fluxes_df, STANDARDIZE=STANDARDIZE_, SCALE_POPFVA=SCALE_POPFVA_)
                pop_sprices_df = ens.scale_df(pop_sprices_df, STANDARDIZE=STANDARDIZE_, SCALE_POPFVA=SCALE_POPFVA_)
                pop_rcosts_df = ens.scale_df(pop_rcosts_df, STANDARDIZE=STANDARDIZE_, SCALE_POPFVA=SCALE_POPFVA_)

                popsol_df_dict = {"pop_fluxes":pop_fluxes_df, "pop_sprices":pop_sprices_df, 
                                  "pop_rcosts":pop_rcosts_df, "pop_sol":pop_sol_df}
                if save_data==True:
                    pop_fluxes_df.to_csv(ENSEMBLE_DIR+"/tables/scaled_fluxes_"+save_file_tag)
                    pop_sprices_df.to_csv(ENSEMBLE_DIR+"/tables/scaled_sprices_"+save_file_tag)
                    pop_rcosts_df.to_csv(ENSEMBLE_DIR+"/tables/scaled_rcosts_"+save_file_tag)
                    pop_sol_df.to_csv(ENSEMBLE_DIR+"/tables/popsol_df_"+save_file_tag)
                    
                popanova_dict = {}
                for popdf_id, popdf in popsol_df_dict.items():
                    X, y = ens.return_filt_matrix(popdf, pheno_to_data2d_dict, pheno_to_Y_dict, pheno_id=pheno_id)
                    popdf_anova_df = ens.compute_ANOVA_test(X, y, correction_test=fdr_correction, correct_alpha=0.05)
                    save_popid_tag = pheno_id+"_"+sample_id+"_OBJdirect-"+obj_direction+"_"+popdf_id+"_"+file_outtag+".csv"
                    
                    if popdf_id=="pop_fluxes" or popdf_id=="pop_rcosts":
                        popdf_anova_df["subsystem"] = popdf_anova_df.index.map(lambda x: SPECIES_MODEL.base_cobra_model.reactions.get_by_id(x).subsystem)
                        popdf_anova_df["name"] = popdf_anova_df.index.map(lambda x: SPECIES_MODEL.base_cobra_model.reactions.get_by_id(x).name)
                        popdf_anova_df["genes"] = popdf_anova_df.index.map(lambda x: [y.id for y in SPECIES_MODEL.base_cobra_model.reactions.get_by_id(x).genes])
                        popdf_anova_df["reaction"] = popdf_anova_df.index.map(lambda x: SPECIES_MODEL.base_cobra_model.reactions.get_by_id(x).reaction)
                        popdf_anova_df["alleles_sim"] = popdf_anova_df.index.map(lambda x: get_alleles_rxn(x))
                        if save_data==True:
                            popdf_anova_df.to_csv(ENSEMBLE_DIR+"/tables/"+save_popid_tag)

                    elif popdf_id=="pop_sprices":
                        popdf_anova_df["name"] = popdf_anova_df.index.map(lambda x: SPECIES_MODEL.base_cobra_model.metabolites.get_by_id(x).name)
                        if save_data==True:
                            popdf_anova_df.to_csv(ENSEMBLE_DIR+"/tables/"+save_popid_tag)

                    elif popdf_id=="pop_sol":
                        if save_data==True:
                            popdf_anova_df.to_csv(ENSEMBLE_DIR+"/tables/"+save_popid_tag)

                    popanova_dict.update({popdf_id: popdf_anova_df})