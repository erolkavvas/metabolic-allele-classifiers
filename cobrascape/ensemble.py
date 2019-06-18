# coding: utf-8
### -------------------------------------------------------------------------
### ensemble.py  
### Erol Kavvas, SBRG, 2018  
### -------------------------------------------------------------------------
###     "ensemble.py" provides a class object for computing with a population 
###     of allele-parameterized strain-specific genome-scale models. The two 
###     optimizations implemented are for population flux variability 
###     analysis (popFVA) and single objective optimization (FBA).
###     Holla at yr homeboi at ekavvas@eng.ucsd.edu if u gots questions dawg.
### --------------------------------------------------------------------------
from tqdm import tqdm
import numpy as np
import pandas as pd
from os import listdir, path
import ast

### Cobra + CobraScape
from cobra.io import load_json_model
from cobrascape.species import load_json_obj, save_json_obj, create_action_set
from sklearn.feature_selection import f_classif
from statsmodels.sandbox.stats.multicomp import multipletests

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import stats
from collections import Counter
from sklearn.utils import Bunch

class Samples(object):
    """ Class object of holding information about the ensemble for analysis of 
        the sampled allele-constraint maps and their resulting popFVA landscapes.
    """
    def __init__(self):
        self.action_num = None
        self.assess_df = pd.DataFrame()
        self.constraint_df = pd.DataFrame()
        self.anova_dict = {}
        self.y_pheno_dict = {}
        self.x_allele_dict = {}
        self.base_cobra_model = None
        self.pheno_list = []
        self.hq_models = pd.DataFrame()
        self.model_weights = pd.DataFrame()
        self.signif_pca = pd.DataFrame()
        self.assess_file_loc = None
        self.popfva_file_loc = None
        self.flux_samples = pd.DataFrame()
        self.gene_to_name = {}

        
        
    def get_hq_samples(self, ic_id="AIC", pheno_id="isoniazid", thresh_val=7):
        """ Returns Series of high quality samples along with the criteria value
        """
        if ic_id == "cv_score_mean":
            max_ic = self.assess_df[ic_id+"_"+pheno_id].max()
            min_ic = max_ic - thresh_val
            hq_models = self.assess_df[ic_id+"_"+pheno_id][self.assess_df[ic_id+"_"+pheno_id]>min_ic]
        else:
            min_ic = self.assess_df[ic_id+"_"+pheno_id].min()
            max_ic = min_ic + thresh_val
            hq_models = self.assess_df[ic_id+"_"+pheno_id][self.assess_df[ic_id+"_"+pheno_id]<max_ic]
        hq_models.sort_values(inplace=True)
        self.hq_models = hq_models
        return hq_models


    def get_ic_weights(self, pheno_id="isoniazid", ic_id="BIC", sample_all=True):
        """ Returns Series of high quality samples along with the criteria value
        """
        if sample_all==True:
            hqmodels_obj = self.get_hq_samples(ic_id=ic_id, pheno_id=pheno_id, thresh_val=2000) # in all samples
        else:
            hqmodels_obj = self.get_hq_samples(ic_id=ic_id, pheno_id=pheno_id, thresh_val=40) # only the top
        hqmodels_obj = hqmodels_obj-min(hqmodels_obj)
        hqmodels_obj_sum = sum([np.exp(x/-2) for x in hqmodels_obj])
        hqmodels_obj = hqmodels_obj.map(lambda x: np.exp(x/-2)/hqmodels_obj_sum)
        self.model_weights = hqmodels_obj
        return hqmodels_obj
    
    
    def popFVA_enriched(self, ic_id="AIC", pheno_id="isoniazid", thresh_val=7, FDR_filter=False):
        """ Mann-whitney U test for whether a particular FVA objective has higher associations
            in high quality models (low AIC) than the other models (non-low AIC)
        """
        if self.hq_models.empty == True:
            self.get_hq_samples(ic_id, pheno_id, thresh_val)
            
        pheno_id = self.hq_models.name.replace("cv_score_mean_","").replace("AIC_","").replace("BIC_","")
        hq_sample_ids = set(self.hq_models.index)
        lq_sample_ids = set(self.assess_df.index) - hq_sample_ids
        s_df = -np.log(self.anova_dict[pheno_id])
        s_df.fillna(1, inplace=True)
        
        feat_to_test_dict = {}
        for feat in s_df.columns:
            if len(s_df[feat].unique())!=1:
                X = s_df.loc[list(hq_sample_ids), feat]
                Y = s_df.loc[list(lq_sample_ids), feat]
                if X.shape[0]!=0:
                    U_stat, pval = stats.mannwhitneyu(X, Y, use_continuity=True, alternative='greater') 
                else:
                    U_stat, pval = 1, 1
                feat_to_test_dict.update({feat: {"stat": U_stat, "pvalue": pval}})
                
        feat_to_test_df = pd.DataFrame.from_dict(feat_to_test_dict, orient="index")
        feat_to_test_df.sort_values("pvalue", ascending=True, inplace=True)
        if FDR_filter == True:
            feat_to_test_df = FDR(feat_to_test_df,fdr_rate=.1)
        return feat_to_test_df
    
    
    def get_logreg_params(self, pval_thresh=5e-2, hq_mods=pd.DataFrame()):
        """ Gets significant PCA components
            - logreg_constant: decides whether the constant variable in logistic regression is considered
        """
        signif_pca_comps, signif_pca_react_loads = {}, {}
        if self.hq_models.empty == True and hq_mods.empty==True:
            print("Give list of models or run -get_hq_samples")
            return None
        elif hq_mods.empty!=True:
            self.hq_models = hq_mods
            
        pheno_id = "_".join(self.hq_models.name.split("_")[1:])
        for sampled_map_num, ic_val in self.hq_models.items():
            landscape_sample_num = sampled_map_num.split("_")[-1]
            sample_id = "sample_"+landscape_sample_num+"_map_assess.json"
            landscape_assess_sample_file = self.assess_file_loc+sample_id
            min_pca_pval=False
            if path.exists(landscape_assess_sample_file):
                landscape_assess_sample = load_json_obj(landscape_assess_sample_file)
                pval_dict = ast.literal_eval(landscape_assess_sample["p_values_"+pheno_id].replace("nan", "1.0"))
                coef_dict = ast.literal_eval(landscape_assess_sample["coefs_"+pheno_id])
                comp_dict = ast.literal_eval(landscape_assess_sample["PCA_comp_dict_"+pheno_id])
                comp_dict = {"x"+str(k+1):v for k, v in comp_dict.items() }
                signif_pca_comps[sampled_map_num] = {}
                for pca_comp, p_val in pval_dict.items():
                    if p_val < pval_thresh and pca_comp!="const":
                        signif_pca_comps[sampled_map_num].update({pca_comp:{
                            "p_val": p_val, "coef": coef_dict[pca_comp],"pca_load":comp_dict[pca_comp]}})
        self.signif_pca = signif_pca_comps
        return signif_pca_comps
                        
        
    def get_sample_pca(self, sample_id, pca_thresh=0.0, pca_comp_id=None, drop_pval=True):
        """ Returns a dataframe of (popFVA features, pca components) for a particular sample.
            pca_thresh: the cutoff value for a popFVA feature in a pca component.
            pca_comp_id: decides which component the dataframe will be sorted by.
        """
        if len(self.signif_pca.keys())==0:
            self.get_logreg_params()

        sample_logreg_df = pd.DataFrame.from_dict(self.signif_pca[sample_id]).T
        sample_logreg_df = sample_logreg_df.sort_values(["p_val"])
        if pca_comp_id==None:
            top_pca_comp = sample_logreg_df.index[0]
        else:
            top_pca_comp = pca_comp_id

        pca_comp_df = pd.DataFrame()
        for pca_comp, pca_row in sample_logreg_df.iterrows():
            comp_df = pd.DataFrame.from_dict(pca_row["pca_load"], orient="index") # orient=pca_row["pca_load"].keys()
            comp_df.columns = [pca_comp]
            pca_comp_df = pd.concat([pca_comp_df, comp_df], axis=1)

        pca_comp_df = pca_comp_df.T
        pca_comp_df["coef"] = pca_comp_df.index.map(lambda x: self.signif_pca[sample_id][x]["coef"])
        pca_comp_df["p_val"] = pca_comp_df.index.map(lambda x: self.signif_pca[sample_id][x]["p_val"])
        pca_comp_df.sort_values(["p_val"], inplace=True)
        pca_comp_df = pca_comp_df.T
        if drop_pval==True:
            pca_comp_df.drop(["coef", "p_val"], axis=0, inplace=True)
        pca_comp_filt = pca_comp_df[abs(pca_comp_df)>pca_thresh]
        pca_comp_filt.dropna(how="all", inplace=True)
        pca_comp_filt.sort_values([top_pca_comp], inplace=True)
        return pca_comp_filt
        
        
    def get_alleles_LOR(self, allele_list, pheno_id="isoniazid", addval=0.5):
        """Takes in a list of alleles and returns the log odds ratio of each allele occurance with a phenotype
        """
        drug_allele_df = filter_0_alleles(self.x_allele_dict[pheno_id].copy())
        LOR_list, num_R_list, num_strains_list, perc_R_list = [], [], [], []
        for x_allele in allele_list:
            strains_with_allele = drug_allele_df[drug_allele_df[x_allele]==1].index.tolist()
            allele_resist_percent = round(resist_percentage(self.y_pheno_dict[pheno_id], strains_with_allele), 4)
            LOR, num_R = log_odds_ratio(x_allele, drug_allele_df, self.y_pheno_dict[pheno_id], addval=addval)
            LOR_list.append(LOR)
            num_R_list.append(num_R)
            num_strains_list.append(len(strains_with_allele))
            perc_R_list.append(allele_resist_percent)
        return LOR_list,num_R_list,num_strains_list,perc_R_list


    def allele_to_rxn_constraints_ids(self, allele_list, action_list=None, samps=None, base_mod=None, allele_gene_sep="_"):
        """ Provides a mapping from alleles to rxns to constraints
            player_list = ["Rv1908c_1", "Rv1484_2", ...]
        """
        ### >>> allele_react_constraint_dict["Rv1908c_1"]
        ### >   {'CAT': {'lb_0': 2.9458181727613184e-06,
        ###              'ub_0': 0.03271365078829508,
        ###              'lb_1': 0.01635829830323392,
        ###              'ub_1': 0.338863211770653}}
        if base_mod==None:
            base_mod = self.base_cobra_model
        if samps==None:
            samps = self.flux_samples
        if action_list==None:
            action_list = create_action_set(number_of_actions=self.action_num, add_no_change=self.add_no_change)

        allele_rxns_constraint_dict = {}
        for all_player in allele_list:
            allele_rxns_constraint_dict[all_player] = {}
            gene_id = all_player.split(allele_gene_sep)[0]
            react_ids = [x.id for x in base_mod.genes.get_by_id(gene_id).reactions]
            for react in react_ids:
                allele_rxns_constraint_dict[all_player][react] = {}
                max_flux, min_flux = max(samps[react]), min(samps[react])
                mean_flux = np.mean(samps[react])
                action_to_constraints_dict = {}
                # for reactions that can't have any change, keep their bounds at a single value.
                if max_flux == min_flux: 
                    for a in action_list:
                        action_to_constraints_dict.update({a: max_flux})
                else:
                    left_bound_distance = mean_flux - min_flux
                    gradient_steps = len(action_list)/2
                    min_to_mean_grad = np.arange(min_flux, mean_flux, (mean_flux-min_flux)/gradient_steps)
                    max_to_mean_grad = np.arange(mean_flux, max_flux, (max_flux-mean_flux)/gradient_steps)
                    for a in action_list:
                        if a == "no_change":
                            action_to_constraints_dict.update({a: 0})
                        else:
                            dec_or_inc = a.split("_")[0]
                            grad_dist = int(a.split("_")[1])
                            # It doesn't matter if mean_flux is less than or greater than 0.
                            if dec_or_inc == "lb": # Change upper_bound
                                action_to_constraints_dict.update({a: min_to_mean_grad[grad_dist]})
                            elif dec_or_inc == "ub": # Change lower_bound
                                action_to_constraints_dict.update({a: max_to_mean_grad[grad_dist]})
                allele_rxns_constraint_dict[all_player][react].update(action_to_constraints_dict)
            
        return allele_rxns_constraint_dict


    def get_correlated_alleles(self,ic_id="BIC",pheno_id="isoniazid",thresh_val=12, pearson=False, gene_name=False,
                               ac_maxmin=False, drop_nochange=False, model_list=[], ac_mapping=None, verbose=False):
        """ Computes correlations amongst allele-constraints in high quality models.
            Function used to be called "get_spearman_alleles()"
            returns top_ac_pq (alleles, alleles) and top_ac_sort_filt (allele, allele, value stacked df)
        """
        if ac_mapping==None:
            action_constraint_mapping = get_action_constraint_mapping(self.action_num, add_no_change=self.add_no_change)
        else:
            action_constraint_mapping = ac_mapping
        if len(model_list)==0:
            self.get_hq_samples(ic_id=ic_id, pheno_id=pheno_id, thresh_val=thresh_val)
            top_ac_df = self.constraint_df.loc[self.hq_models.index.tolist()] #.replace(action_constraint_mapping)
            model_list = self.hq_models.index.tolist()
        else:
            top_ac_df = self.constraint_df.loc[model_list]
            
        if verbose==True:
            print("models #:",len(model_list))
        drug_allele_df = filter_0_alleles(self.x_allele_dict[pheno_id].copy())
        drop_alleles = list(set(top_ac_df.columns) - set(drug_allele_df.columns))
        top_ac_df.drop(drop_alleles, axis=1, inplace=True)

        if gene_name==True:
            top_ac_df = self.convert_gene2name_df(top_ac_df.T.copy())
            top_ac_df = top_ac_df.T

        if drop_nochange==True and self.add_no_change==True:
            top_ac_df = top_ac_df.replace({"no_change":np.nan})
            top_ac_df.dropna(axis=1,how="any",inplace=True)
            
        if ac_maxmin==True:
            top_ac_df = maxmin_allele_df(top_ac_df.columns.tolist(), top_ac_df, genes_list)
        else:
            top_ac_df.replace(action_constraint_mapping, inplace=True)
            
        if pearson==True:
            ac_spearman_rho_df, ac_spearman_pval_df = calculate_pearson_pvalues(top_ac_df)
        else:
            rho, pval = stats.spearmanr(top_ac_df, axis=0)
            ac_spearman_rho_df = pd.DataFrame(rho, index=top_ac_df.columns.tolist(),columns=top_ac_df.columns.tolist())
            ac_spearman_pval_df = pd.DataFrame(pval, index=top_ac_df.columns.tolist(),columns=top_ac_df.columns.tolist())
        
        abs_top_ac_pw = ac_spearman_rho_df #.abs()
        ### Turn correlation matrix into [allele_x, allele_y, corr_val] dataframe
        ac_spearman_rho_df_sort = (abs_top_ac_pw.where(np.triu(np.ones(abs_top_ac_pw.shape), k=1).astype(np.bool)).stack().sort_values(ascending=False))
        ac_spearman_pval_df_sort = (ac_spearman_pval_df.where(np.triu(np.ones(ac_spearman_pval_df.shape), k=1).astype(np.bool)).stack().sort_values(ascending=True))
        return ac_spearman_rho_df, ac_spearman_rho_df_sort, ac_spearman_pval_df,ac_spearman_pval_df_sort


    def convert_gene2name(self, x):
        """ Takes gene id and returns corresponding gene name. self.gene_to_name must be given!
        """
        if not self.gene_to_name: # if gene_to_name dictionary is empty, raise error!
            print("gene_to_name dictionary is empty! returning None...")
            return None
        else:
            if x.split("_")[0] in self.gene_to_name.keys():
                return x.replace(x.split("_")[0], self.gene_to_name[x.split("_")[0]])
            else:
                return x
    
    def convert_gene2name_df(self, input_df):
        """ Takes a dataframe with indices as gene ids and returns a dataframe with gene names
        """
        new_name_dict = {x: self.convert_gene2name(x) for x in input_df.index}
        out_df = input_df.rename(index=new_name_dict).copy()
        return out_df



    def load_ensemble_data(self, STRAIN_NUM=375,ALLELE_NUM=237,ACTION_NUM=4, ADD_NO_CHANGE=False,
                           pheno_list = ["ethambutol", "isoniazid", "rifampicin", "4-aminosalicylic_acid",
                                         "pyrazinamide", "ethionamide","ofloxacin", "cycloserine"],
                           STANDARDIZE=False, FILTER_RXN_DIR=False, test_set=True):
        """ Loads in the data describing a particular ensemble
        """
        self.action_num = ACTION_NUM
        self.add_no_change = ADD_NO_CHANGE
        ENSEMBLE_DIR = "ens_strains"+str(STRAIN_NUM)+"_alleles"+str(ALLELE_NUM)+"_actions"+str(ACTION_NUM)
        if not path.exists(ENSEMBLE_DIR):
            raise ValueError('\t... directory "%s" does not exist' %s (ENSEMBLE_DIR))
        else:
            print("dir ensemble: %s" % (ENSEMBLE_DIR))

        POPFVA_SAMPLES_DIR = ENSEMBLE_DIR+"/popfva_samples/"
        if not path.exists(POPFVA_SAMPLES_DIR):
            print('\t... directory "%s" does not exist' %s (POPFVA_SAMPLES_DIR))
            raise ValueError('\t... directory "%s" does not exist' %s (ENSEMBLE_DIR))
        else:
            print("dir popfva samples: %s" % (POPFVA_SAMPLES_DIR))
        self.popfva_file_loc = POPFVA_SAMPLES_DIR

        ### Create folders to save different types of sample assessments
        if STANDARDIZE==True:
            ENSEMBLE_MAP_ASSESS = ENSEMBLE_DIR+"/popfva_assessment_std/"
            ENSEMBLE_MAP_ANOVA = ENSEMBLE_DIR+"/popfva_anova_std/" ### Save ANOVA F-test enrichments.
        else:
            ENSEMBLE_MAP_ASSESS = ENSEMBLE_DIR+"/popfva_assessment/"
            ENSEMBLE_MAP_ANOVA = ENSEMBLE_DIR+"/popfva_anova/"

        if FILTER_RXN_DIR==True:
            ENSEMBLE_MAP_ASSESS = ENSEMBLE_DIR+"/popfva_assessment_rxnfilt_std"+str(STANDARDIZE)+"/"
            ENSEMBLE_MAP_ANOVA = ENSEMBLE_DIR+"/popfva_anova_rxnfilt_std"+str(STANDARDIZE)+"/"

        self.assess_file_loc = ENSEMBLE_MAP_ASSESS
        ENSEMBLE_MAP_COMPRESS = ENSEMBLE_DIR+"/popfva_compress/" ### Save numpy array versions of landscapes

        ### -------------- LOAD 1 -----------------
        print("(1) load COBRA_MODEL, base_flux_samples, pheno_to_data2d_dict, pheno_to_Y_dict ...")

        MODEL_SAMPLES_FILE = ENSEMBLE_DIR+"/"+"base_flux_samples.csv"
        base_flux_samples = pd.read_csv(MODEL_SAMPLES_FILE,index_col=0)
        self.flux_samples = base_flux_samples

        ENSEMBLE_BASEMODEL_FILE = ENSEMBLE_DIR+"/base_cobra_model.json"
        COBRA_MODEL = load_json_model(ENSEMBLE_BASEMODEL_FILE)
        self.base_cobra_model = COBRA_MODEL

        ### Load in the genetic variant matrix and AMR phenotypes for each case.
        pheno_to_data2d_dict = {}
        pheno_to_Y_dict = {}
        ALLELE_PHENO_FILE = ENSEMBLE_DIR+"/allele_pheno_data/"
        for pheno_id in pheno_list:
            G_VARIANT_MATRIX_FILE = ALLELE_PHENO_FILE+"/allele_df_"+pheno_id+".csv"
            PHENO_MATRIX_FILE = ALLELE_PHENO_FILE+"/pheno_df_"+pheno_id+".csv"
            pheno_to_data2d_dict.update({pheno_id: pd.read_csv(G_VARIANT_MATRIX_FILE,index_col=0)})
            pheno_to_Y_dict.update({pheno_id: pd.read_csv(PHENO_MATRIX_FILE,index_col=0)[pheno_id]})## to make Series
        
        self.x_allele_dict = pheno_to_data2d_dict
        self.y_pheno_dict = pheno_to_Y_dict
        self.pheno_list = pheno_list

        ### -------------- LOAD 2 -----------------
        print("(2) load SAMPLES_ASSESS_DF ...")
        onlyfiles = [f for f in listdir(ENSEMBLE_MAP_ASSESS) if path.isfile(path.join(ENSEMBLE_MAP_ASSESS, f))]
        onlyfiles = [f for f in onlyfiles if f != ".DS_Store"]
        if test_set==True:
            samplesAfter = [f for f in onlyfiles if "sample_" in f][:20]# only get 20 sample so files are small
        else:
            samplesAfter = [f for f in onlyfiles if "sample_" in f] 

        wanted_keys = []
        ### Options for what we want in SAMPLES_ASSESS_DF are as follows... (look in 02_ass_ensemble.py for more info)
        ### "AIC_", "BIC_", "prsquared_", "loglikelihood_", "LLR_pval_", "p_values_", "coefs_", "std_err_", "PCA_comp_dict_"
        for pheno_id in pheno_list:
            wanted_keys.extend(["AIC_"+pheno_id, "BIC_"+pheno_id, "prsquared_"+pheno_id, "std_err_"+pheno_id,
                                "loglikelihood_"+pheno_id, "LLR_pval_"+pheno_id, "cv_score_mean_"+pheno_id])

        SAMPLES_ASSESS_DF = {}
        for landscape_sample_name in tqdm(samplesAfter):
            landscape_sample_num = landscape_sample_name.split("_")[1]
            sample_id = "sampled_map_"+str(landscape_sample_num)
            landscape_assess_sample_file = ENSEMBLE_MAP_ASSESS+landscape_sample_name

            if path.exists(landscape_assess_sample_file):
                landscape_assess_sample = load_json_obj(landscape_assess_sample_file)
                SAMPLES_ASSESS_DF[sample_id] = {}
                SAMPLES_ASSESS_DF[sample_id].update(dict((k, landscape_assess_sample[k]) for k in wanted_keys if k in landscape_assess_sample))

        # transform to pandas dataframe
        SAMPLES_ASSESS_DF = pd.DataFrame.from_dict(SAMPLES_ASSESS_DF,orient="index")
        print("\t... SAMPLES_ASSESS_DF shape: (samples: %d, assess_cols: %d)" % (SAMPLES_ASSESS_DF.shape[0], SAMPLES_ASSESS_DF.shape[1]))
        self.assess_df = SAMPLES_ASSESS_DF

        ### -------------- LOAD 3 -----------------
        print("(3) load SAMPLES_ANOVA_DICT ...")
        SAMPLES_ANOVA_DF = {}
        for pheno_id in pheno_list:
            SAMPLES_ANOVA_DF[pheno_id] = {}

        for landscape_sample_name in tqdm(samplesAfter[:]):
            landscape_sample_num = landscape_sample_name.split("_")[1]
            sample_id = "sample_"+landscape_sample_num+"_map_anova.json"
            landscape_anova_sample_file = ENSEMBLE_MAP_ANOVA+sample_id

            if path.exists(landscape_anova_sample_file):
                landscape_anova_sample = load_json_obj(landscape_anova_sample_file)
                
                for pheno_id in pheno_list:
                    SAMPLES_ANOVA_DF[pheno_id]["sampled_map_"+landscape_sample_num] = {}
                    SAMPLES_ANOVA_DF[pheno_id]["sampled_map_"+landscape_sample_num].update(landscape_anova_sample[pheno_id]["pVal"])

        print("\t... generating SAMPLES_ANOVA_DICT")
        SAMPLES_ANOVA_DICT = {}
        for pheno_id in tqdm(pheno_list):
            SAMPLES_ANOVA_DICT.update({pheno_id: pd.DataFrame.from_dict(SAMPLES_ANOVA_DF[pheno_id],orient="index")})
        
        self.anova_dict = SAMPLES_ANOVA_DICT

        ### -------------- LOAD 3 -----------------
        print("(4) load SAMPLES_AC_DF ...")
        allele_col_ids = [x for x in pheno_to_data2d_dict[pheno_list[0]].columns]

        SAMPLES_AC_DF = {}
        for landscape_sample_name in tqdm(samplesAfter):
            landscape_sample_num = landscape_sample_name.split("_")[1]
            sample_id = "sampled_map_"+str(landscape_sample_num)
            landscape_assess_sample_file = ENSEMBLE_MAP_ASSESS+landscape_sample_name

            if path.exists(landscape_assess_sample_file):
                landscape_assess_sample = load_json_obj(landscape_assess_sample_file)
                SAMPLES_AC_DF[sample_id] = {}
                SAMPLES_AC_DF[sample_id].update(dict((k, landscape_assess_sample[k]) for k in allele_col_ids if k in landscape_assess_sample))

        SAMPLES_AC_DF = pd.DataFrame.from_dict(SAMPLES_AC_DF,orient="index")
        print("\t... SAMPLES_AC_DF shape: (samples: %d, assess_cols: %d)" % (SAMPLES_AC_DF.shape[0], SAMPLES_AC_DF.shape[1]))
        self.constraint_df = SAMPLES_AC_DF
        


def compute_ANOVA_test(X1,y1,correction_test=False,correct_alpha=0.05):
    """ returns ANOVA_test (X1 columns vs f_value, p-value, etc.)
    """
    ANOVA_test = pd.DataFrame(list(f_classif(X1, y1)))
    ANOVA_test.columns = X1.columns
    ANOVA_test = ANOVA_test.sort_values([0, 1], axis=1, ascending=False).T
    ANOVA_test.columns = ["F_value", "pvalue"]
    ANOVA_test["value_counts"] = ANOVA_test.index.map(lambda x: Counter(X1[x].values).most_common())
    if correction_test!=False:
        rejected_list, pvalue_corrected_list, alphaC, alphacBonf = multipletests(
            ANOVA_test["pvalue"], alpha=correct_alpha, method='bonferroni', is_sorted=False)
        ANOVA_test_corrected = ANOVA_test[rejected_list].copy()
        ANOVA_test_corrected["corrected_pVal"] = pvalue_corrected_list[rejected_list]
        return ANOVA_test_corrected
    else:
        return ANOVA_test

        
def FDR(p_values,fdr_rate=.01):
    """False discovery rate boiii
    """
    sorted_vals = p_values.sort_values('pvalue')
    m = len(p_values)
    ranks = range(1,m+1)
    crit_vals = np.true_divide(ranks,m)*fdr_rate
    sig = (sorted_vals.pvalue < crit_vals)
    if len(np.argwhere(sig)) == 0:
        return pd.DataFrame(columns=['log_OR','pvalue','precision','recall','TP'])
    else:
        thresh = np.argwhere(sig)[-1][0]
        final_vals = sorted_vals[:thresh+1]
        return final_vals.sort_values('pvalue',ascending=True)

    
def get_rxn_alleles(rxn, mod, ac_df):
    """plot_alleles = get_rxn_alleles("DCPT", COBRA_MODEL, SAMPLES_AC_DF)
    """
    rxn_gem_obj = mod.reactions.get_by_id(rxn)
    rxn_gem_gene_list = [x.id for x in list(rxn_gem_obj.genes)]
    rxn_alleles = []
    for g_all in ac_df.columns:
        g_ = g_all.split("_")[0]
        if g_ in rxn_gem_gene_list:
            rxn_alleles.append(g_all)
    return rxn_alleles

    
def get_gene_alleles(gene_id, ac_df):
    """plot_alleles = get_gene_alleles("Rv1908c", SAMPLES_AC_DF)
    """
    g_alleles = []
    for g_all in ac_df.columns:
        g_ = g_all.split("_")[0]
        if g_ == gene_id:
            g_alleles.append(g_all)
    return g_alleles


def resist_percentage(resistance_data, list_of_strains):
    return resistance_data.loc[list_of_strains].sum()/float(len(resistance_data.loc[list_of_strains].index))

    
def log_odds_ratio(allele_, allele_df, pheno_df, addval=0.5):
    """Return the log odds ratio of the allele penetrance with the AMR phenotype.
    """
    allele_df["pheno"] = pheno_df
    presence_R = float(len(allele_df[(allele_df[allele_]==1)&(allele_df["pheno"]==1)].index))
    presence_S = float(len(allele_df[(allele_df[allele_]==1)&(allele_df["pheno"]==0)].index))
    absence_R = float(len(allele_df[(allele_df[allele_]==0)&(allele_df["pheno"]==1)].index))
    absence_S = float(len(allele_df[(allele_df[allele_]==0)&(allele_df["pheno"]==0)].index))
    num_R = presence_R
    if presence_R==0 or presence_S==0 or absence_R==0 or absence_S==0:
        presence_R+=addval
        presence_S+=addval
        absence_R+=addval
        absence_S+=addval
        
    odds_ratio = (presence_R/presence_S)/(absence_R/absence_S)
    LOR = np.log(odds_ratio)
    return LOR, num_R


def filter_0_alleles(allele_df, allele_num=2):
    """Drop alleles that do not appear in any of the strains.
    """
    drop_cols = []
    for col in allele_df.columns:
        if allele_df[col].sum()<allele_num:
            drop_cols.append(col)
    allele_df.drop(drop_cols, inplace=True, axis=1)
    return allele_df


def get_action_constraint_mapping(action_number, add_no_change=False):
    """action_constraint_mapping = get_action_constraint_mapping(action_number)
    """
    lb_list = ["lb_"+str(x) for x in range(0, action_number//2, 1)]
    ub_list = ["ub_"+str(x) for x in range(0, action_number//2, 1)]
    if add_no_change==True:
        action_list = lb_list + ["no_change"] + ub_list
        action_ord_list = range(-action_number//2, action_number//2+1)
    else:
        action_list = lb_list + ub_list
        action_ord_list = list(np.arange(-action_number//2, 0))+list(np.arange(1, action_number//2+1))
    action_constraint_mapping = dict(zip(tuple(action_list), tuple(action_ord_list)))
    return action_constraint_mapping


def get_LOR_colors(LOR_list, min_max=(-2, 2)):
    """Use Log Odds Ratios list to create color map for allele columns
    """
    cmap = cm.coolwarm
    if min_max!=False:
        norm = Normalize(vmin=min_max[0], vmax=min_max[1])
    else:
        bnd_Val = max(abs(min(LOR_list)), abs(max(LOR_list)))
        print("min(LOR_list), max(LOR_list): ",min(LOR_list), max(LOR_list))
        norm = Normalize(vmin=-bnd_Val, vmax=bnd_Val)
        
    allele_color_list = [cmap(norm(x)) for x in LOR_list]
    return allele_color_list


def calculate_pearson_pvalues(df):
    """Run pearson to get p-values and correlation coefficients
    """
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    rho = dfcols.transpose().join(dfcols, how='outer')
    for r in tqdm(df.columns):
        for c in df.columns:
            # pvalues[r][c] = round(stats.pearsonr(df[r], df[c])[1], 4)
            rho[r][c], pvalues[r][c] = stats.pearsonr(df[r], df[c])
            # print pvalues[r][c]
    return rho, pvalues


# allele_color_list = get_LOR_colors(LOR_list)
### 2 functions below help out with analyzing interactions amongst constraints
def rxn_to_constraints_samples_ids(player_list, action_list, samps, base_mod):
    """ I should remove reacter
    Parameters
    ----------
    player_list = ["Rv1908c", "Rv1484", ...]
    """

    allele_rxns_constraint_dict = {}
    for all_player in player_list:
        allele_rxns_constraint_dict[all_player] = {}
        react_ids = [x.id for x in base_mod.genes.get_by_id(all_player).reactions]
        for react in react_ids:
            allele_rxns_constraint_dict[all_player][react] = {}
            max_flux, min_flux = max(samps[react]), min(samps[react])
            mean_flux = np.mean(samps[react])
    
            action_to_constraints_dict = {}
            # for reactions that can't have any change, keep their bounds at a single value.
            if max_flux == min_flux: 
                for a in action_list:
                    action_to_constraints_dict.update({a: max_flux})
            else:
                left_bound_distance = mean_flux - min_flux

                gradient_steps = len(action_list)/2
                min_to_mean_grad = np.arange(min_flux, mean_flux, (mean_flux-min_flux)/gradient_steps)
                max_to_mean_grad = np.arange(mean_flux, max_flux, (max_flux-mean_flux)/gradient_steps)

                for a in action_list:
                    if a == "no_change":
                        action_to_constraints_dict.update({a: 0})
                    else:
                        dec_or_inc = a.split("_")[0]
                        grad_dist = int(a.split("_")[1])
                        # It doesn't matter if mean_flux is less than or greater than 0.
                        if dec_or_inc == "lb": # Change upper_bound
                            action_to_constraints_dict.update({a: min_to_mean_grad[grad_dist]})
                        elif dec_or_inc == "ub": # Change lower_bound
                            action_to_constraints_dict.update({a: max_to_mean_grad[grad_dist]})
            allele_rxns_constraint_dict[all_player][react].update(action_to_constraints_dict)
        
    return allele_rxns_constraint_dict



def fva_AMR_clustermap_show(X_AMR_alleles, Y_AMR_pheno, figSIZE=(4, 8), clusterCOL=False,clusterROW=False, save_file=None):
    
    triple_color_palette = [sns.color_palette("RdBu_r", 7, desat=1)[3], sns.color_palette("RdBu_r", 7, desat=1)[-1],
            sns.color_palette("RdBu_r", 7, desat=1)[0]]
    double_color_palette = [sns.color_palette("RdBu_r", 7, desat=1)[3],sns.color_palette("RdBu_r", 7, desat=1)[-1]]
    ### Inputs: INH_species, INH_alleles, INH_phenotype
    specific_color_palette = [sns.color_palette("RdBu_r", 7, desat=1)[0], sns.color_palette("RdBu_r",7, desat=1)[-1]]
    # specific_color_palette = [sns.color_palette("RdBu_r", 3, desat=1)[0], sns.color_palette("RdBu_r", 3, desat=1)[2]]
    X_plot_df = X_AMR_alleles #INH_species.loc[:, INH_alleles]
    Y_plot_df = Y_AMR_pheno.reindex(X_plot_df.index) # INH_phenotype
    
    if True in Y_plot_df[Y_plot_df.columns].isna().any().values:
        Y_plot_df.fillna(2, inplace=True)

    colorsForDF_list = []
    for y_pheno in Y_plot_df.columns:
        labels = Y_plot_df[y_pheno].values
        if len(Y_plot_df[y_pheno].unique())>2:
            lut = {0.0: (0.9690888119953865, 0.9664744329104191, 0.9649365628604382),
             1.0: (0.7284890426758939, 0.15501730103806222, 0.19738562091503264), 
             2.0: (0.7614763552479814, 0.8685121107266436, 0.924567474048443)}
            # lut = dict(zip(set(labels), triple_color_palette))
        else:
            lut = {0.0: (0.9690888119953865, 0.9664744329104191, 0.9649365628604382),
             1.0: (0.7284890426758939, 0.15501730103806222, 0.19738562091503264)}
            # lut = dict(zip(set(labels), double_color_palette))
        # lut = dict(zip(set(labels), specific_color_palette)) # sns.hls_palette(len(set(labels)), l=0.5, s=0.8))
        row_colors_iter = pd.DataFrame(labels)[0].map(lut)
        colorsForDF_list.append(row_colors_iter)
    ### Drop FVA columns that have no differences (i.e., all the same values)
    drop_cols = []
    for col in X_plot_df.columns:
        if len(X_plot_df[col].unique())==1:
            drop_cols.append(col)
    X_plot_df.drop(drop_cols, axis=1, inplace=True)

    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    g=sns.clustermap(abs(X_plot_df), 
                     # method='average', metric='euclidean',
                     method='ward', metric='euclidean', # metric='correlation',
                     standard_scale=True, # z_score=True,
                     row_cluster=clusterROW, col_cluster=clusterCOL,
                     row_colors=colorsForDF_list,cmap=cmap,figsize=figSIZE);
    
    if save_file!=None:
        g.ax_heatmap.set_title(save_file.split("/")[-1])
        g.savefig(save_file+".png", dpi=150);
        g.savefig(save_file+".svg");
    return g


def scale_df(popfva_ls, STANDARDIZE=False, SCALE_POPFVA=True):
    """ Scales the provided dataframe using either StandardScaler (True, Z-score)
        or MinMaxScaler (False, normalization) implemented in sklearn. Scale popfva decides whether 
        to scale or not. If not, returns the input popfva_ls without any changes.
    """
    if STANDARDIZE==True:
        landscape_scaler = StandardScaler() # Standardization Z-score
    else:
        landscape_scaler = MinMaxScaler() # Normalization 0-1 scaling

    if SCALE_POPFVA==True:
        popfva_ls_scaled = landscape_scaler.fit_transform(popfva_ls)
        X_df= pd.DataFrame(popfva_ls_scaled,index=popfva_ls.index, columns=popfva_ls.columns)
    else:
        X_df = popfva_ls
    return X_df


def return_filt_matrix(X_, x_allele_dict, y_pheno_dict, pheno_id="isoniazid"):
    """ Return X, y matrices using dictionaries directly instead of ensemble Sample object.
        Useful when constructing large Sample object is not desired.
    """
    X_ = X_.reindex(x_allele_dict[pheno_id].index)
    Y_ = y_pheno_dict[pheno_id].reindex(x_allele_dict[pheno_id].index)
    return X_, Y_


def filter_amr_fva(fva_landscape_df, G_ALLELE_clustermap_data2d, Y_pheno_test):
    """Notice the use of MinMaxScaler below.
    """
    Y_pheno_test_reindexed = Y_pheno_test.reindex(G_ALLELE_clustermap_data2d.index)
    fva_landscape_df_reindexed = fva_landscape_df.reindex(G_ALLELE_clustermap_data2d.index)
    ### --- 
    landscape_scaler = MinMaxScaler() # StandardScaler()
    fva_landscape_df_reindexed.fillna(0, inplace=True)
    G_FVA_clustermap_scaled = landscape_scaler.fit_transform(fva_landscape_df_reindexed)
    G_FVA_clustermap = pd.DataFrame(G_FVA_clustermap_scaled, 
                                    index=fva_landscape_df_reindexed.index, 
                                    columns=fva_landscape_df_reindexed.columns)
    X_standardscaled_SAMPLE = G_FVA_clustermap.reindex(G_ALLELE_clustermap_data2d.index)
    y = Y_pheno_test_reindexed.astype(int)
    y.dropna(axis=0, how='all', inplace=True)
    X = X_standardscaled_SAMPLE
    X = X.reindex(y.index)
    return X, y
    

def load_landscape_sample(fva_landscape_file):
    """Load the popFVA landscape for a particular model sample
    """
    fva_landscape_dict = load_json_obj(fva_landscape_file)
    obj_val_list = {}
    for strain_id, strain_fva_dict in fva_landscape_dict.items():
        obj_val_list[strain_id] = {}
        for rxn, max_min_dict in strain_fva_dict.items():
            obj_val_list[strain_id].update({rxn+"_max":float(format(max_min_dict["maximum"],'.10f')), 
                                            rxn+"_min":float(format(max_min_dict["minimum"],'.10f'))})
    fva_landscape_df = pd.DataFrame.from_dict(obj_val_list, orient="index")
    return fva_landscape_df


def get_sample_constraints(variant_dec_file):
    """Load the allele-constraint map for a particular model sample"
    """
    if path.exists(variant_dec_file):
        variant_dec_dict = load_json_obj(variant_dec_file)
    else:
        print("variant_dec_dict does not exist: ", variant_dec_file)
        variant_dec_dict = {}
    return variant_dec_dict


def get_rxn_list(popfva_ls, cobra_model):
    rxn_obj_list = []
    for rxn_obj in popfva_ls.columns:
        if "_max" in rxn_obj:
            rxn_id = rxn_obj.split("_max")[0]
        elif "_min" in rxn_obj:
            rxn_id = rxn_obj.split("_min")[0]
        rxn = cobra_model.reactions.get_by_id(rxn_id)
        if rxn.reversibility == False:
            if rxn.lower_bound>=0:
                #print(rxn.reaction, rxn.id+"_max")
                rxn_obj_list.append(rxn.id+"_max")
            elif rxn.lower_bound<0:
                #print(rxn.reaction, rxn.id+"_min")
                rxn_obj_list.append(rxn.id+"_min")
        else:
            rxn_obj_list.append(rxn.id+"_min")
            rxn_obj_list.append(rxn.id+"_max")
            
    rxn_obj_list = list(set(rxn_obj_list))
    return rxn_obj_list
    # print(len(popfva_ls.columns),len(rxn_obj_list))


def get_allele_set(s_obj, pval__df,rho__df, popfva_ml_df=pd.DataFrame(), pheno_id="isoniazid", return_type="pval",
                   pval_cutoff=0.05, topmodelnum=5, rxn_gems=None, gene_name=False, verbose=False):
    """ Read in Sample object and pval, rho dataframes. Return a list of alleles satisfying the function's parameters.
    """
    if popfva_ml_df.empty==True:
        print("no popfva_ml_df passed... setting gene_popfva_list=[]")
        gene_popfva_list = []
    else:
        gene_popfva_list = []
        for gpr in popfva_feat_df["gpr"].values:
            gene_popfva_list.extend(ast.literal_eval(gpr))
        gene_popfva_list = list(set(gene_popfva_list))
        if gene_name==True:
            gene_popfva_list = [s_obj.gene_to_name[x] if x in s_obj.gene_to_name.keys() else x for x in gene_popfva_list]
        # popfva_feat_df = pd.read_csv("supplement/Supplementary File 3/"+amr_drug+"_permut_importance_TOPMODELNUM-"+str(topmodelnum)+"_rf.csv", index_col=0)

    pval_df_filt = pval__df[(pval__df<pval_cutoff)&(abs(rho__df)<0.9999)].copy()
    pval_df_filt.dropna(axis=0, how="all", inplace=True)
    pval_df_filt.dropna(axis=1, how="all", inplace=True)
    pval_genes = list(set([x.split("_")[0] for x in pval_df_filt.index.tolist()]))
    pval_popfva_genes =list(set(pval_genes).intersection(set(gene_popfva_list)))
    pval_popfva_alleles = []
    for pval_allele in pval_df_filt.index.tolist():
        if pval_allele.split("_")[0] in pval_popfva_genes:
            pval_popfva_alleles.append(pval_allele)

    drug_allele_df = filter_0_alleles(s_obj.x_allele_dict[pheno_id].copy())
    if rxn_gems!=None:
        plot_alleles = []
        for rxn_gem in rxn_gems:
            plot_alleles.extend(get_rxn_alleles(rxn_gem, s_obj.base_cobra_model, drug_allele_df))
        pval_react_alleles = list(set(pval_df_filt.index.tolist()).intersection(set(plot_alleles)))
    else:
        pval_react_alleles = []

    if verbose==True:
        print("\tlen(gene_popfva_list): ", len(gene_popfva_list))
        print("\tlen(pval_df_filt.index): ", len(pval_df_filt.index))
        print("\tlen(pval_genes): ", len(pval_genes))
        print("\tlen(pval_popfva_genes): ",len(pval_popfva_genes))
        print("\tlen(pval_popfva_alleles): ",len(pval_popfva_alleles))
        print("\tlen(pval_react_alleles): ",len(pval_react_alleles))

    ### Specify which set of alleles to return
    return_var=None
    if return_type=="pval":
        print("returning pval alleles ... ", len(pval_df_filt.index))
        return_var = pval_df_filt.index.tolist()
    elif return_type=="react":
        print("returning react alleles ... ", len(pval_react_alleles))
        return_var = pval_react_alleles
    elif return_type=="popfva":
        print("returning popfva alleles... ", len(pval_popfva_alleles))
        return_var = pval_popfva_alleles
    return return_var



def get_rxn_list(popfva_ls_df, cobra_model):
    rxn_obj_list = []
    for rxn_obj in popfva_ls_df.columns:
        if "_max" in rxn_obj:
            rxn_id = rxn_obj.split("_max")[0]
        elif "_min" in rxn_obj:
            rxn_id = rxn_obj.split("_min")[0]
        rxn = cobra_model.reactions.get_by_id(rxn_id)
        if rxn.reversibility == False:
            if rxn.lower_bound>=0:
                #print(rxn.reaction, rxn.id+"_max")
                rxn_obj_list.append(rxn.id+"_max")
            elif rxn.lower_bound<0:
                #print(rxn.reaction, rxn.id+"_min")
                rxn_obj_list.append(rxn.id+"_min")
        else:
            rxn_obj_list.append(rxn.id+"_min")
            rxn_obj_list.append(rxn.id+"_max")
            
    rxn_obj_list = list(set(rxn_obj_list))
    return rxn_obj_list




    
