import seaborn as sns
import matplotlib.pyplot as plt
from random import shuffle
import pandas as pd
from sklearn.metrics import roc_curve, auc
import os
from matplotlib import cm
import numpy as np
from tqdm import tqdm
from scipy import stats
from adjustText import adjust_text
import math
import itertools

from collections import Counter
from statsmodels.sandbox.stats.multicomp import multipletests
from scipy.stats import hypergeom
import ast
import cobrascape.ensemble as ens

def set_sample_pheno_popobj(ensemble_dir, species_mod, sample_id, obj_direct, pheno_id="isoniazid", ic_id="BIC", 
                       ic_rank=0, pval_threshold=1, load_threshold=0):
    PCA_DATA_DIR = ensemble_dir+"/pca_data/"
    sample_id_num = sample_id.split("_")[-1]
    PCA_SAMPLE_FN = PCA_DATA_DIR+"obj_sampled_map_"+sample_id_num+"__"+pheno_id+".json"
    print(PCA_SAMPLE_FN)
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
    print("...set PCA objective")
    species_mod = cs.set_linear_popfva_objective(species_mod, r_filt_df, obj_dir=obj_direct)
    return species_mod, sample_ac, r_filt_df



def get_mnc_direction(pheno_id, sample_id, ENSEMBLE_DIR):
    """ Return whether the MNC optimization is a minimization or maximization
    """
    popdf_id = "pop_sol"
    obj_direct_pval_list = []
    for obj_direction in ["max", "min"]:
        found_file=False
        popsol_anova_loc = ENSEMBLE_DIR+"/tables/"+pheno_id+"_"+sample_id+"_OBJdirect-"+obj_direction+"_"+"pop_sol"+"_train.csv"
        if os.path.exists(popsol_anova_loc):
            popsol_anova_df = pd.read_csv(popsol_anova_loc, index_col=0)
        else:
            print("can't locate %s"%(popsol_anova_loc))
        obj_direct_pval_list.append((obj_direction, popsol_anova_df["pvalue"][0]))
    srt = sorted(obj_direct_pval_list,key=lambda x: x[1], reverse=True)
    obj_direction = srt[-1][0]
    return obj_direction, obj_direct_pval_list

# def get_mnc_direction(pheno_id, sample_id, ENSEMBLE_DIR, ic_rank=True):
#     """ Return whether the MNC optimization is a minimization or maximization
#     """
#     popdf_id = "pop_sol"
#     obj_direct_pval_list = []
#     for obj_direction in ["max", "min"]:
#         found_file=False
#         for i in range(30):
#             popsol_anova_loc = ENSEMBLE_DIR+"/tables/"+pheno_id+"_"+sample_id+"_OBJdirect-"+obj_direction+"_PCArank_"+str(i)+"_"+"pop_sol"+"_train.csv"
#             if os.path.exists(popsol_anova_loc):
#                 popsol_anova_df = pd.read_csv(popsol_anova_loc, index_col=0)
#                 found_file=True
#                 break
#         if found_file==False:
#             print("can't locate %s"%(popsol_anova_loc))
#         obj_direct_pval_list.append((obj_direction, popsol_anova_df["pvalue"][0]))
#     srt = sorted(obj_direct_pval_list,key=lambda x: x[1], reverse=True)
#     obj_direction = srt[-1][0]
#     return obj_direction, obj_direct_pval_list




def load_sample_pheno_data(pheno_id, sample_id, obj_direction, ENSEMBLE_DIR, scale=True, ic_rank=True):
    """ Load data given phenotype, MNC sample, MNC optimization direction
        ic_rank is a useless parameter that I can't get rid of due to being too deep bro :(
    """
    if ic_rank==True:
        for ic_rank in range(20):
            popsol_anova_loc = ENSEMBLE_DIR+"/tables/"+pheno_id+"_"+sample_id+"_OBJdirect-"+obj_direction+"_PCArank_"+str(ic_rank)+"_"+"pop_sol"+"_train.csv"
            if os.path.exists(popsol_anova_loc):
                ### Load raw computations
                sample_sol_dict, sample_anova_dict = {}, {}
                if scale==True:
                    pop_fluxes_df=pd.read_csv(ENSEMBLE_DIR+"/tables/scaled_fluxes_"+pheno_id+"_"+sample_id+"_OBJdirect-"+obj_direction+"_PCArank_"+str(ic_rank)+"_train.csv",index_col=0)
                    pop_sprices_df=pd.read_csv(ENSEMBLE_DIR+"/tables/scaled_sprices_"+pheno_id+"_"+sample_id+"_OBJdirect-"+obj_direction+"_PCArank_"+str(ic_rank)+"_train.csv",index_col=0)
                    pop_rcosts_df=pd.read_csv(ENSEMBLE_DIR+"/tables/scaled_rcosts_"+pheno_id+"_"+sample_id+"_OBJdirect-"+obj_direction+"_PCArank_"+str(ic_rank)+"_train.csv",index_col=0)
                else:
                    pop_fluxes_df=pd.read_csv(ENSEMBLE_DIR+"/tables/sol_fluxes_"+pheno_id+"_"+sample_id+"_OBJdirect-"+obj_direction+"_PCArank_"+str(ic_rank)+"_train.csv",index_col=0)
                    pop_sprices_df=pd.read_csv(ENSEMBLE_DIR+"/tables/sol_sprices_"+pheno_id+"_"+sample_id+"_OBJdirect-"+obj_direction+"_PCArank_"+str(ic_rank)+"_train.csv",index_col=0)
                    pop_rcosts_df=pd.read_csv(ENSEMBLE_DIR+"/tables/sol_rcosts_"+pheno_id+"_"+sample_id+"_OBJdirect-"+obj_direction+"_PCArank_"+str(ic_rank)+"_train.csv",index_col=0)
                pop_sol_df=pd.read_csv(ENSEMBLE_DIR+"/tables/popsol_df_"+pheno_id+"_"+sample_id+"_OBJdirect-"+obj_direction+"_PCArank_"+str(ic_rank)+"_train.csv",index_col=0)
                sample_sol_dict = {
                    "pop_fluxes": pop_fluxes_df,
                    "pop_rcosts": pop_rcosts_df,
                    "pop_sprices": pop_sprices_df,
                    "pop_sol": pop_sol_df,
                }
                ### Load ANOVA association data
                for popdf_id in ["pop_fluxes", "pop_rcosts", "pop_sprices", "pop_sol"]:
                    popdf_anova_df = pd.read_csv(ENSEMBLE_DIR+"/tables/"+pheno_id+"_"+sample_id+"_OBJdirect-"+obj_direction+"_PCArank_"+str(ic_rank)+"_"+popdf_id+"_train.csv",index_col=0)
                    sample_anova_dict.update({popdf_id: popdf_anova_df})
                break
    else:
        save_file_tag = pheno_id+"_"+sample_id+"_OBJdirect-"+obj_direction+"_train.csv"
        #save_popid_tag = pheno_id+"_"+sample_id+"_OBJdirect-"+obj_direction+"_"+popdf_id+"_train.csv"
        popsol_anova_loc = ENSEMBLE_DIR+"/tables/"+pheno_id+"_"+sample_id+"_OBJdirect-"+obj_direction+"_"+"pop_sol"+"_train.csv"
        if os.path.exists(popsol_anova_loc):
            sample_sol_dict, sample_anova_dict = {}, {}
            if scale==True:
                pop_fluxes_df=pd.read_csv(ENSEMBLE_DIR+"/tables/scaled_fluxes_"+save_file_tag,index_col=0)
                pop_sprices_df=pd.read_csv(ENSEMBLE_DIR+"/tables/scaled_sprices_"+save_file_tag,index_col=0)
                pop_rcosts_df=pd.read_csv(ENSEMBLE_DIR+"/tables/scaled_rcosts_"+save_file_tag,index_col=0)
            else:
                pop_fluxes_df=pd.read_csv(ENSEMBLE_DIR+"/tables/raw_fluxes_"+save_file_tag,index_col=0)
                pop_sprices_df=pd.read_csv(ENSEMBLE_DIR+"/tables/raw_sprices_"+save_file_tag,index_col=0)
                pop_rcosts_df=pd.read_csv(ENSEMBLE_DIR+"/tables/raw_rcosts_"+save_file_tag,index_col=0)
            pop_sol_df=pd.read_csv(ENSEMBLE_DIR+"/tables/popsol_df_"+save_file_tag,index_col=0)
            sample_sol_dict = {
                "pop_fluxes": pop_fluxes_df,
                "pop_rcosts": pop_rcosts_df,
                "pop_sprices": pop_sprices_df,
                "pop_sol": pop_sol_df,
            }
            ### Load ANOVA association data
            for popdf_id in ["pop_fluxes", "pop_rcosts", "pop_sprices", "pop_sol"]:
                save_popid_tag = pheno_id+"_"+sample_id+"_OBJdirect-"+obj_direction+"_"+popdf_id+"_train.csv"
                popdf_anova_df = pd.read_csv(ENSEMBLE_DIR+"/tables/"+save_popid_tag,index_col=0)
                sample_anova_dict.update({popdf_id: popdf_anova_df})
        else:
            print(popsol_anova_loc, "does not exist!")
    
    return sample_sol_dict, sample_anova_dict


def get_roc_auc(fpr_, tpr_, roc_auc_, sample_sol_, pheno_id, y_pheno_dict):
    ### Get AMR phenotypes for test data
    y = y_pheno_dict[pheno_id].copy()
    popsol_df_decision = sample_sol_.copy()
    popsol_df_decision[y.name+"_predict"] = popsol_df_decision["sol"]
    
    # Compute ROC curve and ROC area for each class
    y.replace(2, 1, inplace=True)
    y_test = y.copy()
    y_score = popsol_df_decision[y.name+"_predict"].copy()
    y_score = y_score.reindex(y_test.index)
    y_test_array = np.asarray(y_test)
    y_score_array = np.asarray(y_score)
    fpr_[pheno_id], tpr_[pheno_id], _ = roc_curve(y_test_array, y_score_array)
    roc_auc_[pheno_id] = auc(fpr_[pheno_id], tpr_[pheno_id])
    if roc_auc_[pheno_id]<0.50:
        roc_auc_[pheno_id] = 1-roc_auc_[pheno_id] # This means the negative sign does the opposite
    return fpr_, tpr_, roc_auc_



def correct_pvals(X_df, pval_col="pvalue", method="bonferroni", correct_alpha=0.05):
    """ performs multiple-testing correction on the list of pvalues
    """
    rejected_list, pvalue_corrected_list, alphaC, alphacBonf = multipletests(
        X_df[pval_col], alpha=correct_alpha, method=method, is_sorted=False)
    ANOVA_test_corrected = X_df[rejected_list].copy()
    ANOVA_test_corrected["corrected_pVal"] = pvalue_corrected_list[rejected_list]
    return ANOVA_test_corrected


def get_avg_df(pheno_pval_s_dict, pheno_id, avg_type="mean"):
    if avg_type=="mean":
        med_pval_df = pheno_pval_s_dict[pheno_id].fillna(1.0)
        med_pval_df = med_pval_df.replace(np.inf, 1.0e-100)
        med_pval_df = med_pval_df.replace(0, 1.0e-2)
        # med_pval_df.replace(np.inf, 1.0e-100, inplace=True)
        # med_pval_df= med_pval_df+1e-100
        # med_pval_df.replace(-np.inf, 0.0, inplace=True)
        # med_pval_df= med_pval_df+1e-20
        med_pval_df = -np.log10(med_pval_df)
        med_pval_df[avg_type] = med_pval_df.apply(lambda x: x.mean(), axis=1)
    elif avg_type=="median":
        med_pval_df = pheno_pval_s_dict[pheno_id].fillna(1.0)
        # med_pval_df= med_pval_df+1e-20
        med_pval_df.replace(np.inf, 0.0, inplace=True)
        med_pval_df = -np.log10(med_pval_df)
        # med_pval_df.replace(-np.inf, 0.0, inplace=True)
        med_pval_df[avg_type] = med_pval_df.apply(lambda x: x.median(), axis=1)
    med_pval_df.sort_values([avg_type], ascending=False, inplace=True)
    return med_pval_df


### Plots for annotated manhatten
def plot_setup(row_num=1, col_num=1, s_size=100, labelsizes=20, f_scale=1.0, figSIZE=(7,5), sharex=False,sharey=False):
    """Returns f, ax"""
    rc_par = {"axes.labelsize": labelsizes, "xtick.labelsize":labelsizes, 
              "ytick.labelsize":labelsizes,"axes.titlesize":labelsizes}
    # savefig=True
    with sns.plotting_context("notebook", font_scale=f_scale, rc=rc_par):
        f, ax = plt.subplots(row_num, col_num, figsize=figSIZE, sharex=sharex,sharey=sharey)
        return f, ax
    
def plot_manhatten(X_df, y_id, ax=None, hue=None, palette="Set1", s_size=40,kwargs=None):
    if ax==None:
        f, ax = plot_setup(row_num=1, col_num=1, s_size=100, labelsizes=20, f_scale=1.0, figSIZE=(7,5))
    x = [i for i in range(len(X_df.index))]
    shuffle(x)
    X_df["rand_index"] = x
    ax = sns.scatterplot(x="rand_index", y=y_id, data=X_df, s=s_size,
                         hue=hue, palette=palette, color="k",ax=ax,**kwargs) # , s=s_size
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
    return X_df, ax


def annotate_plot(X_df, y_col, x_col, ax, threshold=30, fontsize=12):
    texts = []
    expand_args = {'expand_objects':(1.2,1.4),'expand_points':(1.3,1.3), 'expand_text':(1.4,1.4)}
    # expand_args['expand_text'] =(1.4,1.4)
    # print(expand_args)
    for feat, feat_row in X_df.iterrows():
        y_feat, x_feat = feat_row[y_col], feat_row[x_col]
        # if abs(c_feat)>load_threshold:
        if abs(y_feat)>threshold:
            # texts.append(ax_obj_hist.text(c_feat, 4, popfva_feat, fontsize=12))
            texts.append(ax.text(x_feat, y_feat, feat, fontsize=fontsize))
            # ax_obj_hist.annotate(popfva_feat, xy=(c_feat, 15/(abs(c_feat))+2), rotation=30, color="k")
            # arrowprops=dict(facecolor='black', shrink=0.05))
    adjust_text(texts,ax=ax, arrowprops=dict(arrowstyle="-",color='k',lw=0.5),
                only_move={'objects':'xy'},**expand_args)
    return ax


def get_sig_feat_df(ENSEMBLE_DIR, pheno_id, sample_list, y_pheno_dict, feat_type="pop_fluxes",med_or_mean="mean",ic_rank=False):
    pheno_list = [pheno_id]
    scale_ = True
    samples_flux_anova_df = pd.DataFrame()
    sample_pheno_dict = {}
    for sample_map in tqdm(sample_list):
        sample_id = "sample_"+sample_map.split("_")[-1]
        fpr_, tpr_, roc_auc_ = {}, {}, {} 
        sample_pheno_dict[sample_id] = {}
        for pheno_id in pheno_list:
            obj_direction, _ = get_mnc_direction(pheno_id, sample_id, ENSEMBLE_DIR)
            # print(obj_direction, _)
            s_sol_dict, s_anova_dict = load_sample_pheno_data(pheno_id,sample_id,obj_direction,ENSEMBLE_DIR,scale=scale_,ic_rank=ic_rank)
            fpr_, tpr_, roc_auc_ = get_roc_auc(fpr_, tpr_, roc_auc_, s_sol_dict["pop_sol"], pheno_id, y_pheno_dict)
            sample_pheno_dict[sample_id].update({pheno_id: {"sol": s_sol_dict, "anova": s_anova_dict}})

    name_df = sample_pheno_dict[sample_id][pheno_id]["anova"]["pop_fluxes"]
    metab_name_df = sample_pheno_dict[sample_id][pheno_id]["anova"]["pop_sprices"]
    
    pheno_pval_s_dict = {}
    for pheno_id in pheno_list[:]:
        samples_flux_anova_df = pd.DataFrame()
        for sample_map in sample_list:
            sample_id = "sample_"+sample_map.split("_")[-1] # "pop_fluxes"
            sol_anova_pvals = sample_pheno_dict[sample_id][pheno_id]["anova"][feat_type]["pvalue"]
            sol_anova_pvals.name = sample_id
            samples_flux_anova_df = pd.concat([samples_flux_anova_df, pd.DataFrame(sol_anova_pvals)],axis=1,sort=False)
        pheno_pval_s_dict.update({pheno_id: samples_flux_anova_df})
    
    if med_or_mean=="mean":
        sig_flux_df = get_avg_df(pheno_pval_s_dict, pheno_id, avg_type="mean") # "mean","median"
    elif med_or_mean=="median":
        sig_flux_df = get_avg_df(pheno_pval_s_dict, pheno_id, avg_type="median")
    if feat_type=="pop_sprices":
        for id_name in ["name"]:
            sig_flux_df[id_name] = sig_flux_df.index.map(lambda x: metab_name_df.loc[x, id_name])
    else:
        for id_name in ["subsystem","name","alleles_sim", "genes","reaction"]:
            sig_flux_df[id_name] = sig_flux_df.index.map(lambda x: name_df.loc[x, id_name])
            
    return sig_flux_df, sample_pheno_dict



def plot_flux_boxplot(ENSEMBLE_DIR, sample_id, popfva_df, react_plot_list, pheno_id, obj_direction,pre_y=False,
                      savefig=True,whis=1.5,s_size=100,width=0.5,labelsizes=10,notch=True,jitter=True,linewidth=1,
                      dodge=True,palette=["#2020FF", "#FF0303"],f_scale=1.0,figSIZE=(40,5),plot_type="boxplot",
                      alpha=1,edgecolor="k",ax=None):
    """
    plot_type: boxplot, stripplot, violin, swarm
    """
    pop_id = "pop_fluxes"
    rc_par = {"axes.labelsize": labelsizes, "xtick.labelsize":labelsizes, 
              "ytick.labelsize":labelsizes,"axes.titlesize":labelsizes}
    with sns.plotting_context("notebook", font_scale=f_scale, rc=rc_par):
        if ax==None:
            f_bp, ax_bp = plt.subplots(1, figsize=figSIZE)
        else:
            ax_bp = ax
        if pre_y==False:
            X, y = ens.return_filt_matrix(popfva_df, pheno_to_data2d_dict, pheno_to_Y_dict, pheno_id=pheno_id)
            X_y_df = pd.concat([X,y],axis=1)
            X_y_df.reset_index(inplace=True)
        else:
            X_y_df = popfva_df
        X_y_df_melt = pd.melt(X_y_df, id_vars=['index', pheno_id], 
                              value_vars=react_plot_list, 
                              var_name="reaction",value_name='flux')
        
        if plot_type=="violin":
            ax_bp = sns.violinplot(x="reaction", y="flux",data=X_y_df_melt, width=width,
                            palette=["#2020FF", "#FF0303"], hue=pheno_id, scale="count")
        elif plot_type=="boxplot":
            ax_bp = sns.boxplot(x="reaction", y="flux",data=X_y_df_melt, width=width,showmeans=True,
                                showfliers=True,palette=["#2020FF", "#FF0303"], hue=pheno_id,
                                whis=whis,notch=notch,meanline=True,ax=ax_bp)
            
        elif plot_type=="stripplot":
            ax_bp = sns.stripplot(x="reaction", y="flux",data=X_y_df_melt,jitter=jitter, 
                                  size=s_size, alpha=alpha,linewidth=linewidth,
                                  edgecolor=edgecolor,dodge=dodge,palette=palette, hue=pheno_id,ax=ax_bp)
            
#         if plot_type=="violin":
#             ax_bp = sns.violinplot(x=x_id, y=y_id,data=X_y_df_melt, width=width,
#                             palette=palette, hue=pheno_id, scale="count")
#         elif plot_type=="boxplot":
#             ax_bp = sns.boxplot(x=x_id, y=y_id,data=X_y_df_melt, width=width,showmeans=True,
#                                 showfliers=True,palette=palette, hue=pheno_id,
#                                 whis=whis,notch=notch,meanline=True,ax=ax_bp)
            
#         elif plot_type=="swarm":
#             ax_bp = sns.swarmplot(x=x_id, y=y_id,data=X_y_df_melt, size=s_size,alpha=alpha,
#                                   marker=strip_marker,linewidth=linewidth,edgecolor=edgecolor,dodge=dodge,
#                                   hue_order=hue_order_list,#edgecolor="k",linewidth=0.5,
#                                   palette=palette, hue=hue_type,ax=ax_bp)
            
#         elif plot_type=="stripplot":
#             ax_bp = sns.stripplot(x="reaction", y="flux",data=X_y_df_melt,jitter=jitter,hue_order=hue_order_list, 
#                                   size=s_size, alpha=alpha,marker=strip_marker,linewidth=linewidth,
#                                   edgecolor=edgecolor,dodge=dodge,palette=palette, hue=hue_type,ax=ax_bp)
        # ax_bp.legend(False)
        ax_bp.set_title(sample_id+"-"+obj_direction)
        if savefig==True:
            sample_id = sample_id.replace("/", "_")
            sample_id = sample_id.replace(", ", "_")
            ax_bp.figure.savefig(ENSEMBLE_DIR+"/figures/scale_fluxes_boxplot_"+pheno_id+"_"+sample_id+"_OBJdirect-"+obj_direction+"_train.png")
            ax_bp.figure.savefig(ENSEMBLE_DIR+"/figures/scale_fluxes_boxplot_"+pheno_id+"_"+sample_id+"_OBJdirect-"+obj_direction+"_train.svg")
        return ax_bp
            
def plot_fva_boxplot(ENSEMBLE_DIR, sample_id, popfva_df, react_plot_list, pheno_id, obj_direction, 
                     FVA_FRAC_OPTIMUM, savefig=True,whis=1.5,s_size=100,labelsizes=10,f_scale=1.0,figSIZE=(40,5)):
    pop_id = "pop_fluxes"
    rc_par = {"axes.labelsize": labelsizes, "xtick.labelsize":labelsizes, 
              "ytick.labelsize":labelsizes,"axes.titlesize":labelsizes}
    with sns.plotting_context("notebook", font_scale=f_scale, rc=rc_par):
        f_bp, ax_bp = plt.subplots(1, figsize=figSIZE)
        X_y_df_melt = pd.melt(popfva_df, id_vars=['index','optdir','vdir', pheno_id], 
                              value_vars=react_plot_list, 
                              var_name="reaction",value_name='flux')
        ax_bp = sns.boxplot(x="reaction", y="flux",data=X_y_df_melt, width=0.5,showmeans=True,showfliers=True,
                            palette=["#2020FF", "#FF0303"], hue=pheno_id,whis=whis,notch=True,meanline=True,ax=ax_bp)
        ax_bp.set_title(sample_id+" OBJdirect-"+obj_direction+", fvafrac-"+str(FVA_FRAC_OPTIMUM))
        if savefig==True:
            sample_id = sample_id.replace("/", "_")
            sample_id = sample_id.replace(", ", "_")
            f_bp.savefig(ENSEMBLE_DIR+"/figures/fva_boxplot_"+sample_id+"_"+pheno_id+"_OBJdirect-"+obj_direction+"_fvafrac-"+str(FVA_FRAC_OPTIMUM)+"_train.png")
            f_bp.savefig(ENSEMBLE_DIR+"/figures/fva_boxplot_"+sample_id+"_"+pheno_id+"_OBJdirect-"+obj_direction+"_fvafrac-"+str(FVA_FRAC_OPTIMUM)+"_train.svg")
            

def get_flux_df_anova(flux_df, pheno_id, name_df):
    # flux_df.dropna(inplace=True)
    popdf_anova_df = ens.compute_ANOVA_test(
            flux_df[flux_df.columns[:-2]], flux_df[pheno_id], 
            correction_test=False, correct_alpha=0.05)

    for id_name in ["subsystem","name","alleles_sim", "genes","reaction"]:
        popdf_anova_df[id_name] = popdf_anova_df.index.map(lambda x: name_df.loc[x, id_name])
    return popdf_anova_df


def get_sprice_df_anova(sprice_df, pheno_id, name_df):
    popdf_anova_df = ens.compute_ANOVA_test(
            sprice_df[sprice_df.columns[:-2]], sprice_df[pheno_id], 
            correction_test=False, correct_alpha=0.05)

    for id_name in ["name"]:
        popdf_anova_df[id_name] = popdf_anova_df.index.map(lambda x: name_df.loc[x, id_name])
    return popdf_anova_df


def get_subsys_pval_thresh(popdf_anova_df,pheno_id, ENSEMBLE_DIR, verbose=False, savefig=True):
    pvals, reacts = [], []
    X_df = popdf_anova_df
    for pval_thresh in np.logspace(-2, -8, 50):
    # for pval_thresh in np.linspace(5e-2, 5e-10, 100):
        X_sig_df = popdf_anova_df[popdf_anova_df["pvalue"]<pval_thresh].copy()
        if verbose==True:
            print(pval_thresh, X_sig_df.shape)
        pvals.append(pval_thresh)
        reacts.append(X_sig_df.shape[0])

    fig, ax_subsystem_pvalthresh = plot_setup(row_num=1, col_num=1, s_size=100, labelsizes=15, f_scale=1.0, figSIZE=(7,5))
    ax_subsystem_pvalthresh.scatter(pvals, reacts)
    ax_subsystem_pvalthresh.set_xlabel("P-values")
    ax_subsystem_pvalthresh.set_ylabel("Number of significant reactions")
    ax_subsystem_pvalthresh.set_title(pheno_id)
    pval_thresh_df = pd.DataFrame([pvals, reacts],index=["pvals", "reacts"]).T
    pval_thresh_df.plot("pvals", "reacts", kind="scatter", logx=True, ax=ax_subsystem_pvalthresh)
    if savefig==True:
        fig.savefig(ENSEMBLE_DIR+"/figures/subsys_pvalthresh_"+pheno_id+"_train.png")
        fig.savefig(ENSEMBLE_DIR+"/figures/subsys_pvalthresh_"+pheno_id+"_train.svg")
    num_sig_react_diff = []
    for i in range(len(reacts)-1):
        num_sig_react_diff.append(reacts[i]-reacts[i+1])

    best_pval_thresh_ind = np.argmax(num_sig_react_diff)
    best_pval_thresh = (pvals[best_pval_thresh_ind]+pvals[best_pval_thresh_ind+1])/2
    if verbose==True:
        print(reacts[best_pval_thresh_ind], reacts[best_pval_thresh_ind+1])
        print(num_sig_react_diff[best_pval_thresh_ind], best_pval_thresh)
    return best_pval_thresh


def get_subsystem_enrichments(popdf_anova_df, pheno_id, med_or_med="mean",pval_thresh=0.05, save_data=True):
    popdf_anova_df.loc[["TMDS", "TMDS3"], "subsystem"] = "Folate Metabolism"
    # popdf_anova_df.loc[["NNAM", "TMDS3"], "subsystem"] = "Folate Metabolism"
    # pval_thresh = pheno_subsys_pval[pheno_id] #4e-2#1.0e-3#1.8e-2
    X_df = popdf_anova_df
    X_sig_df = popdf_anova_df[popdf_anova_df["pvalue"]<pval_thresh]
    print(pval_thresh, X_sig_df.shape)
    subsys_pval_dict = {}
    for subsys_id in X_df["subsystem"].unique():
        M_total = X_df["subsystem"].values.tolist()
        n_total = X_df["subsystem"][X_df["subsystem"]==subsys_id].values.tolist()
        k_ = X_sig_df["subsystem"][X_sig_df["subsystem"]==subsys_id].values.tolist()
        N_total = len(X_sig_df.index)
        sf_ = hypergeom.sf(len(k_), len(M_total), len(n_total), N_total)
        subsys_pval_dict.update({subsys_id: sf_})
        # print(subsys_id, sf_)

    X_sig_df["allele_sim_bin"] = X_sig_df["alleles_sim"].map(lambda x: 0 if isinstance(x, str) else 1)
    print("\t", len(X_sig_df["allele_sim_bin"][X_sig_df["allele_sim_bin"]==0]))
    print("\t", len(X_sig_df["allele_sim_bin"][X_sig_df["allele_sim_bin"]==1]))
    print("\t", len(X_sig_df["allele_sim_bin"].index))
    print("\t", len(X_sig_df["allele_sim_bin"][X_sig_df["allele_sim_bin"]==1])/len(X_sig_df["allele_sim_bin"].index))

    subsys_sig = pd.DataFrame.from_dict(subsys_pval_dict,orient="index")
    subsys_sig.columns = ["subsys_pval_"+pheno_id]
    subsys_sig.sort_values(["subsys_pval_"+pheno_id], inplace=True)
    if save_data==True:
        if med_or_med=="mean":
            subsys_sig.to_csv(ENSEMBLE_DIR+"/tables/hq_mean_flux_subsystems_"+pheno_id+".csv")
        elif med_or_med=="median":
            subsys_sig.to_csv(ENSEMBLE_DIR+"/tables/hq_median_flux_subsystems_"+pheno_id+".csv")
    return subsys_sig


def get_pathway_enrichments(ENSEMBLE_DIR, popdf_anova_df, pheno_id, gene_to_pathways, kegg_biocyc="biocyc", med_or_med="mean",pval_thresh=0.05, save_data=True):
    # popdf_anova_df.loc[["TMDS", "TMDS3"], "subsystem"] = "Folate Metabolism"
    react_to_pathways = {}
    TOTAL_SUBSYSTEMS = []
    for react, row in popdf_anova_df[:].iterrows():
        react_to_pathways[react] = []
        react_gene_list = ast.literal_eval(row["genes"])
        # print(react, row["genes"])
        # print(react_gene_list)
        if len(react_gene_list)!=0:
            for g in react_gene_list:
                if g in gene_to_pathways.keys():
                    react_to_pathways[react].extend(gene_to_pathways[g])
        react_to_pathways[react] = list(set([x for x in react_to_pathways[react] if x!=""]))
        TOTAL_SUBSYSTEMS.extend(react_to_pathways[react])
        
    UNIQUE_SUBSYSTEMS = list(set(TOTAL_SUBSYSTEMS))
    
    X_df = popdf_anova_df
    X_sig_df = popdf_anova_df[popdf_anova_df["pvalue"]<pval_thresh]
    SIG_SUBSYSTEMS = []
    for react, row in X_sig_df[:].iterrows():
        SIG_SUBSYSTEMS.extend(react_to_pathways[react])
    
    print(pval_thresh, X_sig_df.shape)
    subsys_pval_dict = {}
    
    
    M_total = len(TOTAL_SUBSYSTEMS)
    for subsys_id in UNIQUE_SUBSYSTEMS:
        n_total = TOTAL_SUBSYSTEMS.count(subsys_id)
        k_ = SIG_SUBSYSTEMS.count(subsys_id)
        N_total = len(SIG_SUBSYSTEMS)
        sf_ = hypergeom.sf(k_, M_total, n_total, N_total)
        subsys_pval_dict.update({
            subsys_id: {"subsys_pval_"+pheno_id: sf_,
                        "SIG_SUBSYS_NUM": k_,
                        "TOTAL_SUBSYS_NUM": n_total
                                            }})
    
    subsys_sig_react_dict = {}
    for subsys_id in list(set(SIG_SUBSYSTEMS)):
        subsys_sig_react_dict[subsys_id] = []
        for react_id, react_subsystems in X_sig_df[:].iterrows():
            if subsys_id in react_to_pathways[react_id]:
                subsys_sig_react_dict[subsys_id].append(react_id)

    X_sig_df["allele_sim_bin"] = X_sig_df["alleles_sim"].map(lambda x: 0 if isinstance(x, str) else 1)
    print("\t", len(X_sig_df["allele_sim_bin"][X_sig_df["allele_sim_bin"]==0]))
    print("\t", len(X_sig_df["allele_sim_bin"][X_sig_df["allele_sim_bin"]==1]))
    print("\t", len(X_sig_df["allele_sim_bin"].index))
    print("\t", len(X_sig_df["allele_sim_bin"][X_sig_df["allele_sim_bin"]==1])/len(X_sig_df["allele_sim_bin"].index))

    subsys_sig = pd.DataFrame.from_dict(subsys_pval_dict,orient="index")
    # subsys_sig.rename(mapper={x: x.replace("<i>", "").replace("</i>","") for x in subsys_sig.index},inplace=True)
    # subsys_sig.rename(mapper={x: x.replace("&", "").replace(";","") for x in subsys_sig.index},inplace=True)
    # subsys_sig.columns = ["subsys_pval_"+pheno_id]
    subsys_sig.sort_values(["subsys_pval_"+pheno_id], inplace=True)
    if save_data==True:
        if kegg_biocyc=="biocyc":
            out_id = pheno_id+"_biocyc"
        elif kegg_biocyc=="kegg":
            out_id = pheno_id+"_kegg"
        else:
            out_id = pheno_id
        if med_or_med=="mean":
            subsys_sig.to_csv(ENSEMBLE_DIR+"/tables/hq_mean_flux_BIOCYC-subsystems_"+out_id+".csv")
        elif med_or_med=="median":
            subsys_sig.to_csv(ENSEMBLE_DIR+"/tables/hq_median_flux_BIOCYC-subsystems_"+out_id+".csv")
    return subsys_sig, react_to_pathways, subsys_sig_react_dict


def get_subsys_phenos_df(df, pval_drop=0.005, add_0pval=1e-3, qval=False):
    drop_rows = []
    for subsys, row in df.iterrows():
        if row.min()>pval_drop: #0.005
            drop_rows.append(subsys)
              
    df.drop(drop_rows,inplace=True)
    df = df + add_0pval
    if qval==True:
        df = df.apply(lambda x: qvalue.estimate(qvalue.estimate(x.values)), axis=0)
    
    df.rename(mapper={x: x.replace("<i>", "").replace("</i>","") for x in df.index},inplace=True)
    df.rename(mapper={x: x.replace("&", "").replace(";","") for x in df.index},inplace=True)
    df.rename(mapper={x: x.replace("<sup>", "").replace("</sup>","") for x in df.index},inplace=True)
    return df

##### ----------------------------------------------------------------
##### Code for plotting median bound allele on pathway flux boxplots
##### ----------------------------------------------------------------

from matplotlib.colors import Normalize
from operator import itemgetter
import sklearn
import statsmodels #.api as sm
from sklearn import tree

# gene_to_name = ens.load_json_obj("cobra_model/gene_to_name.json")

def func_convert_gene2name(x, gene_to_name):
        """ Takes gene id and returns corresponding gene name. self.gene_to_name must be given!
        """
        if x.split("_")[0] in gene_to_name.keys():
            return x.replace(x.split("_")[0], gene_to_name[x.split("_")[0]])
        else:
            return x
    
def func_convert_gene2name_df(input_df, gene_to_name):
    """ Takes a dataframe with indices as gene ids and returns a dataframe with gene names
    """
    new_name_dict = {x: func_convert_gene2name(x, gene_to_name) for x in input_df.index}
    out_df = input_df.rename(index=new_name_dict).copy()
    return out_df


def get_allele_data(allele_list, allele_df, y_pheno_dict, 
                    pheno_id="isoniazid", cmap=cm.bwr, cmap_norm = Normalize(vmin=-4, vmax=4),
                    order_alleles="lor", reverse=True, verbose=False):
    """
    Input: list of alleles
    Output: ordered list of allele datatypes
    
    Params:
        order_alleles: "lor", "strain_num", None
    NOTE: allele_list ids need to match those in allele_df!
    """
    # two lists allow for specific ordering of alleles by LOR or strain num
    allele_lor, allele_strain_num = [],[] 
    allele_lor_color_dict = {}
    for allele in allele_list:
        num_strains_with_allele = len(allele_df[allele_df[allele]==1].index.tolist())
        LOR, num_R = ens.log_odds_ratio(allele, allele_df, y_pheno_dict[pheno_id], addval=0.5)
        
        allele_lor.append((allele, LOR))
        allele_strain_num.append((allele, num_strains_with_allele))
        allele_lor_color_dict.update({allele: cmap(cmap_norm(LOR))})
        if verbose==True:
            print("%s: LOR=%s, strain_num=%s"%(allele, LOR, num_strains_with_allele))
    
    allele_list_ordered = []
    if order_alleles=="lor":
        allele_lor.sort(key=itemgetter(1), reverse=reverse)
        allele_list_ordered = [x[0] for x in allele_lor]
    elif order_alleles=="strain_num":
        allele_strain_num.sort(key=itemgetter(1), reverse=reverse)
        allele_list_ordered = [x[0] for x in allele_strain_num]
    elif order_alleles==None:
        allele_list_ordered = allele_list
        
    # allele_colors can be used as a palette
    allele_colors = [allele_lor_color_dict[x] for x in allele_list_ordered]
    return allele_list_ordered, allele_colors
    

# allele_order, palette_plot = get_allele_data(
#     alleles_trace_list, drug_allele_df, pheno_to_Y_dict, pheno_id="pyrazinamide", 
#     cmap=cm.bwr, cmap_norm = Normalize(vmin=-4, vmax=4), order_alleles="lor", reverse=True, verbose=False
# )


def plot_flux_boxplot_alleles(x_id, y_id, alleles, data_df, react_plot_list, pheno_id, ENSEMBLE_DIR,
                               obj_direction, pheno_to_data2d_dict, hue_type="alleles",pre_y=False, alpha=1,
                              jitter=True,strip_marker=None,
                              savefig=True,whis=1.5,s_size=100,width=0.5,labelsizes=10,notch=True,dodge=True,
                              linewidth=1,edgecolor="k",boxplot_background=True,legend=False,
                              f_scale=1.0,figSIZE=(40,5),ylabel_visible=True, y_pheno_dict=None,
                              palette=["#2020FF", "#FF0303"],plot_type="boxplot",ax=None):
    """
        hue_type: "alleles", pheno_id
        plot type : [boxplot, swarm, violin]
    """
    pop_id = "pop_fluxes"
    rc_par = {"axes.labelsize": labelsizes, "xtick.labelsize":labelsizes, 
              "ytick.labelsize":labelsizes,"axes.titlesize":labelsizes, "axes.fontname": "Arial"}
    with sns.plotting_context("notebook", font_scale=f_scale, rc=rc_par):
        if ax==None:
            f_bp, ax_bp = plt.subplots(1, figsize=figSIZE)
        else:
            ax_bp = ax
        if pre_y==False:
            X, y = ens.return_filt_matrix(data_df, pheno_to_data2d_dict, y_pheno_dict, pheno_id=pheno_id)
            X_y_df = pd.concat([X,y],axis=1)
            X_y_df.reset_index(inplace=True)
        else:
            X_y_df = data_df
            drug_allele_df = ens.filter_0_alleles(pheno_to_data2d_dict[pheno_id].copy(),allele_num=0)
            drug_allele_df = func_convert_gene2name_df(drug_allele_df.T, gene_to_name)
            drug_allele_df = drug_allele_df.T
            drug_allele_df["alleles"] = "other"
            
            for allele in alleles:
                strains = drug_allele_df[drug_allele_df[allele]==1].index.tolist()
                drug_allele_df.loc[strains, "alleles"] = allele
                x = drug_allele_df.index[0]
            X_y_df = pd.concat([X_y_df,drug_allele_df["alleles"]],axis=1,sort=True)
            
            susceptible_strains = X_y_df[X_y_df[pheno_id]==0].index.tolist()
            X_y_df.loc[susceptible_strains, pheno_id]= "susceptible" # "S"
            
            resistant_strains = X_y_df[X_y_df[pheno_id]==1].index.tolist() 
            X_y_df.loc[resistant_strains, pheno_id]= "resistant" # "R"
            
            X_y_df.sort_values([pheno_id], inplace=True)
            
        
        id_vars_list = ['index', pheno_id, "alleles"]
        X_y_df_melt = pd.melt(X_y_df, id_vars=id_vars_list, #['index', pheno_id], 
                              value_vars=react_plot_list, 
                              var_name="reaction",value_name='flux')
        if hue_type=="alleles":
            try:
                if "other" in drug_allele_df["alleles"].unique():
                    hue_order_list = alleles + ["other"]
                    palette = palette + ["#d9d9d9"] # grey
                else:
                    hue_order_list = alleles
            except:
                hue_order_list = X_y_df[hue_type].unique()
        elif hue_type==pheno_id:
            hue_order_list = X_y_df[hue_type].unique()
            # hue_order_list = drug_allele_df["alleles"].unique()
        
        if plot_type=="violin":
            ax_bp = sns.violinplot(x=x_id, y=y_id,data=X_y_df_melt, width=width,
                            palette=palette, hue=pheno_id, scale="count")
        elif plot_type=="boxplot":
            ax_bp = sns.boxplot(x=x_id, y=y_id,data=X_y_df_melt, width=width,showmeans=True,
                                showfliers=True,palette=palette, hue=pheno_id,
                                whis=whis,notch=notch,meanline=True,ax=ax_bp)
            
        elif plot_type=="swarm":
            ax_bp = sns.swarmplot(x=x_id, y=y_id,data=X_y_df_melt, size=s_size,alpha=alpha,
                                  marker=strip_marker,linewidth=linewidth,edgecolor=edgecolor,dodge=dodge,
                                  hue_order=hue_order_list,#edgecolor="k",linewidth=0.5,
                                  palette=palette, hue=hue_type,ax=ax_bp)
            
        elif plot_type=="stripplot":
            ax_bp = sns.stripplot(x=x_id, y=y_id,data=X_y_df_melt,jitter=jitter,hue_order=hue_order_list, 
                                  size=s_size, alpha=alpha,marker=strip_marker,linewidth=linewidth,
                                  edgecolor=edgecolor,dodge=dodge,palette=palette, hue=hue_type,ax=ax_bp)
        ax_bp.set_title("--".join(react_plot_list))
        
        # ax_boxplot.xaxis.label.set_visible(False)
        # ax_boxplot.title.set_visible(False)
        
        if boxplot_background==True:
            hue_order_boxplots = ["resistant", "susceptible"]
            ax_bp = sns.boxplot(x=pheno_id, y="flux",data=X_y_df_melt, hue=pheno_id,#hue="alleles",
                                hue_order=hue_order_boxplots, 
                                meanline=True,
                                showfliers=False,palette=["#2020FF", "#FF0303"], dodge=False,
                                showmeans=True,
                                width=width,
                                whis=whis,notch=notch,ax=ax_bp,
                                # kwargs = {"positions": range(1,3)},
                               meanprops={"color":"k", "linestyle":"-", "linewidth":1},
                                #whiskerprops={"color":"k", "linestyle":"-", "linewidth":1},
                                # boxprops = {"linewidth":1},
                               # boxprops = {"linewidth":1},
                               ) # meanline=True,
            
            for patch in ax_bp.artists:
                r, g, b, a = patch.get_facecolor()
                patch.set_facecolor((r, g, b, 0.00)) #  0.05
            # ax_bp.get_legend().remove()
            # ax_bp.legend(False)
            
        if legend==False:
            ax_bp.get_legend().remove()
        if ylabel_visible==False:
            ax_bp.yaxis.label.set_visible(False)
        
        if savefig==True:
            sample_id = sample_id.replace("/", "_")
            sample_id = sample_id.replace(", ", "_")
            ax_bp.figure.savefig(ENSEMBLE_DIR+"/figures/scale_fluxes_boxplot_"+pheno_id+"_"+sample_id+"_OBJdirect-"+obj_direction+"_train.png")
            ax_bp.figure.savefig(ENSEMBLE_DIR+"/figures/scale_fluxes_boxplot_"+pheno_id+"_"+sample_id+"_OBJdirect-"+obj_direction+"_train.svg")

        
        return ax_bp, X_y_df, X_y_df_melt

    
    
def label_allele_constraint(pheno_id, react, allele_to_med_bound, allele_to_category_bound, allele_df, y_pheno_dict, 
                            annotate_type="dots",left_right="right",scaler_vh="h",y_vscaler=0.05,
                            x_text_origin=1.53,x_stack_scaler=1.05,x_text_scaler=0.12,constraint_lw=2,
                            cmap=cm.bwr,norm_swarm=Normalize(vmin=-2, vmax=2),
                            cmap_category=cm.BrBG,norm_category=Normalize(vmin=-2, vmax=2),
                            train_num_thresh=3, verbose=False, ax=None):
    
    if ax==None:
        f_bp, ax_bp = plt.subplots(1, figsize=figSIZE)
    else:
        ax_bp = ax
        
    text_list, allele_bnd_list = [], []
#     ### Get list of all the bounds so we can stack the alleles based on vertical location
#     for allele, bnd in allele_to_med_bound[react].items():
#         allele_bnd_list.append(bnd)
        
    allele_bnd_list = []
    for allele, bnd in allele_to_med_bound[react].items():
        num_strains_with_allele = len(allele_df[allele_df[allele]==1].index.tolist())
        LOR, num_R = ens.log_odds_ratio(allele, allele_df, y_pheno_dict[pheno_id], addval=0.5)
        allele_label = allele #+ " (%s)"%(str(num_strains_with_allele))
        allele_letter_num = len(allele_label)
        allele_color_add = allele_letter_num*x_text_scaler
        # for adding dots instead of coloring...
        if annotate_type=="dots":
            # text_dist_allele = allele_letter_num/100.0
            text_dist = x_stack_scaler*allele_bnd_list.count(bnd)
        else:
            text_dist = 1.2*allele_bnd_list.count(bnd)
            
        if scaler_vh=="v":
            text_dist = x_stack_scaler
            bnd_add=y_vscaler*allele_bnd_list.count(bnd)
        else:
            bnd_add=0
        
        if left_right=="left":
            text_name = -x_text_origin-text_dist
            text_dot = -x_text_origin+allele_color_add-text_dist
            
        elif left_right=="right":
            text_name = x_text_origin+text_dist
            text_dot = x_text_origin+allele_color_add+text_dist
            
            
        if num_strains_with_allele>=strain_num_thresh:
            allele_bnd_list.append(bnd)
            if verbose==True:
                print(allele, bnd, LOR, allele_to_category_bound[allele], num_strains_with_allele)
            allele_color = cmap(norm_swarm(LOR))# cmap(norm(LOR))
            allele_category_color = cmap_category(norm_category(allele_to_category_bound[allele]))
            ax_bp.axhline(
                y=bnd, color=allele_category_color, linestyle='-', linewidth=constraint_lw, alpha=0.9
            )
            if annotate_type=="dots":
                ax_bp.text(
                    text_name, bnd+bnd_add, allele_label, color="k",va="center",style='italic',
                    rotation=0,size=labelsizes, alpha=1
                )
                ax_bp.text(
                    text_dot,bnd+bnd_add, "o", color=allele_color,va="center",style='italic', 
                    rotation=0,size=4, alpha=1, fontname='Arial',
                    bbox=dict(boxstyle="circle",facecolor=allele_color, edgecolor="k",linewidth=1,pad=1,alpha=1.0)
                )
            else:
                ax_bp.text(
                    text_name,bnd+bnd_add, allele_label, color='k',va="center",style='italic',
                    rotation=0,size=labelsizes, alpha=1,fontname='Arial',
                    bbox=dict(facecolor=allele_color, edgecolor=allele_color,linewidth=1, pad=4, alpha=1.0)
                )
    return ax_bp



# cmap = cm.bwr
# norm_swarm = Normalize(vmin=-4, vmax=4)
# cmap_category = cm.BrBG
# norm_category = Normalize(vmin=-2, vmax=2)

# react = "NNAM"# "NNAM"#"ASNt2r"
# ax_boxplot, Xy_df, Xy_df_melt = plot_flux_boxplot_alleles(
#     "reaction", "flux", allele_order,flux_df,[react], pheno_id, ENSEMBLE_DIR,  
#     med_or_mean, pheno_to_data2d_dict,hue_type="alleles",#"alleles",
#     pre_y=True, whis=1.5, width=width,savefig=False,s_size=8,labelsizes=labelsizes,
#     notch=False, f_scale=1.0,figSIZE=figsize,palette=palette_plot,strip_marker="o",#"s",
#     plot_type=plot_type_set,linewidth=0.3,edgecolor="k",alpha=1.0,dodge=True,jitter=0.3,
#     boxplot_background=True, ax=None
# )

# ax_boxplot = label_allele_constraint(
#     pheno_id, react, allele_to_med_bound, allele_to_category_bound,drug_allele_df, pheno_to_Y_dict, 
#     annotate_type="dots",scaler_vh="v",y_vscaler=0.08,
#     left_right="left",x_text_origin=1.8, x_stack_scaler=1.2,x_text_scaler=0.125,
#     # left_right="right",x_text_origin=1.52, x_stack_scaler=1.02,x_text_scaler=0.125,
#     cmap=cmap,norm_swarm=norm_swarm,cmap_category=cmap_category,norm_category=norm_category,
#     train_num_thresh=3, ax=ax_boxplot
# )