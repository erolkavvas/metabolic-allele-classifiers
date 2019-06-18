### CobraScape: a package for constraint-based genome-scale modeling of genetic variation.
# - Classes: Species, Strain, Allele
# - Methods: build_strain_specific() - strain-specific genome-scale reconstructions

from __future__ import absolute_import
from tqdm import tqdm
from copy import copy, deepcopy
from six import iteritems, iterkeys, string_types

# import cobra
from cobra.core.dictlist import DictList
from cobra.util.context import HistoryManager, resettable, get_context
from cobra.core.object import Object

from functools import partial
import pandas as pd
from multiprocessing import cpu_count


class Species(Object):
    """Class representation of Species object
    
    Parameters
    ----------
    species_model_id : Species, string
        Either an existing Species object in which case a new model object is
        instantiated with the same properties as the original model,
        or a the identifier to associate with the model as a string.
    name : string
        Human readable name for the Species model
    Attributes
    ----------
    strains : DictList
        A DictList where the key is the strain identifier and the value a Strain
    alleles : DictList
        A DictList where the key is the allele identifier and the value an Allele
    base_cobra_model : COBRA Model or None
        A cobra Model object
    phenotypes: pandas dataframe
        A mapping of strains in species to observed phenotypic trait 
        indices = strains, columns=traits, values = 0, 1, NaN.
        Where 1 represents the measured presence of the trait and 0 
        represents the measured absence of the trait.
    """
    
    def __setstate__(self, state):
        """Make sure all cobra.Objects in the model point to the model.
        """
        self.__dict__.update(state)
        for y in ['strains', 'alleles', 'base_cobra_model']:
            for x in getattr(self, y):
                x._model = self
        if not hasattr(self, "name"):
            self.name = None
    
    def __getstate__(self):
        """Get state for serialization.
        Ensures that the context stack is cleared prior to serialization,
        since partial functions cannot be pickled reliably.
        """
        odict = self.__dict__.copy()
        odict['_contexts'] = []
        return odict
        
    def __init__(self, species_model_id=None, name=None):
        if isinstance(species_model_id, Species):
            Object.__init__(self, name=name)
        else:
            Object.__init__(self, species_model_id, name=name)
            self.alleles = DictList() # A list of Species.alleles
            self.strains = DictList() # A list of Species.strains
            # self.genes = DictList() # A list of Species.genes
            self.base_cobra_model = None # Base genome-scale model of species
            # self.solution = [] # Base genome-scale model of species (only used in sequential strain simulation)
            # self.reactions
            # self.phenotypes = DictList()
            self.strain_allele_matrix = None
            self.strain_pheno_matrix = None
            self._contexts = []
        
    def load_from_matrix(self, strain_allele_df, filter_model_genes=True, allele_gene_sep=None):
        """Initializes the Species object using strain vs allele matrix
        Parameters
        ----------
        strain_allele_df : A pandas dataframe of strains (index) vs alleles (columns)
                           where the values are binary (0=absence, 1=presence)
        filter_model_genes : Determines whether `filter_cobra_model_genes(strain_allele_df)` will 
                             run (Default=True) 
        allele_gene_sep : separator type for the unique extraction of the GEM gene id from the allele id
                          (Default="_")
        """
        
        def filter_cobra_model_genes():
            """Drops the alleles (columns) from Species vs Alleles matrix that are not in 
            the genome-scale model.
            Parameters:
            -----------
            """
            model_genes = [str(x.id) for x in self.base_cobra_model.genes]
            keep_alleles = []
            for col in strain_allele_df.columns:
                if col.split(allele_gene_sep)[0] in model_genes:
                    keep_alleles.append(col)
            # X_model = strain_allele_df.drop(drop_alleles, axis=1)
            print("# alleles:", len(strain_allele_df.columns), "-> removing alleles not in GEM -> # alleles after:", len(keep_alleles))
            return keep_alleles
        
        
        if filter_model_genes==True and allele_gene_sep!=None and self.base_cobra_model!=None:
            df_alleles = filter_cobra_model_genes()
        else:
            df_alleles = strain_allele_df.columns
        
        strains_to_add = [Strain(x) for x in strain_allele_df.index]
        alleles_to_add = [Allele(x) for x in df_alleles]
        self.add_strains(strains_to_add)
        self.add_alleles(alleles_to_add)
        self.strain_allele_matrix=strain_allele_df

        for strain in tqdm(self.strains):
            strain.add_alleles(dict(strain_allele_df.loc[strain.id, df_alleles]))
            
    # def load_measured_phenotypes(self, strain_traits_df, filter_NaN_traits=False):
    #     """Initializes the phenotypes part of the Species object using strain vs traits matrix
    #     Parameters
    #     ----------
    #     strain_phenotype_df : A pandas dataframe of strains (index) vs traits (columns)
    #                        where the values are binary (0=absence, 1=presence)
    #     filter_model_genes : Determines whether `filter_cobra_model_genes(strain_allele_df)` will
    #                          run (Default=True)
    #     allele_gene_sep : separator type for the unique extraction of the GEM gene id from the allele id
    #                       (Default="_")
    #     """
    #
    #     # def filter_cobra_model_genes():
    #     #     """Drops the alleles (columns) from Species vs Alleles matrix that are not in
    #     #     the genome-scale model.
    #     #     Parameters:
    #     #     -----------
    #     #     """
    #     #     model_genes = [str(x.id) for x in self.base_cobra_model.genes]
    #     #     keep_alleles = []
    #     #     for col in strain_allele_df.columns:
    #     #         if col.split(allele_gene_sep)[0] in model_genes:
    #     #             keep_alleles.append(col)
    #     #     # X_model = strain_allele_df.drop(drop_alleles, axis=1)
    #     #     print "...Filtering alleles || # alleles before:", len(strain_allele_df.columns),
    #     #     print "| # alleles after:", len(keep_alleles)
    #     #     return keep_alleles
    #
    #     if filter_model_genes==True and allele_gene_sep!=None and self.base_cobra_model!=None:
    #         df_alleles = filter_cobra_model_genes()
    #     else:
    #         df_alleles = strain_allele_df.columns
    #
    #     strains_to_add = [Strain(x) for x in strain_allele_df.index]
    #     alleles_to_add = [Allele(x) for x in df_alleles]
    #     self.add_strains(strains_to_add)
    #     self.add_alleles(alleles_to_add)
    #
    #     for strain in tqdm(self.strains):
    #         strain.add_alleles(dict(strain_allele_df.loc[strain.id, df_alleles]))
        
        
    def update_strains_cobra_model(self, remove_absent=False, verbose=False):
        """Creates a unique copy of the base cobra model for each strain
            Make sure that the base cobra model solver is set to "gplk" for max speed.
        """
        if self.base_cobra_model:
            if verbose==True:
                print("\t... Attach a unique base cobra model copy to each strain (may take a while)")
            gene_ids = [str(x.id) for x in self.base_cobra_model.genes]
            for strain in tqdm(self.strains):
                strain._cobra_model = self.base_cobra_model.copy()
                
                for allele in strain._alleles:
                    if allele._cobra_gene: # check if a cobra gene exists
                        if allele._cobra_gene in gene_ids:
                            # allele_reacts_list = {}
                            for allele_react in strain._cobra_model.genes.get_by_id(allele._cobra_gene).reactions:
                                if allele_react.id in allele._cobra_reactions.keys():
                                    allele._cobra_reactions[allele_react.id].append(allele_react)
                                else:
                                    allele._cobra_reactions.update({allele_react.id: [allele_react]})
                                # allele_reacts_list.append(allele_react)
                            # allele._cobra_reactions.extend(allele_reacts_list)
        else:
            raise ValueError('No base cobra model found in  {}')

    
    def optimize_strains(self, fva_rxn_set="var_reacts", parallel=True, fva=False, fract_opt=0.1, processes=cpu_count()):
        """Will add a list of alleles to the model object and add new
        constraints accordingly. Alleles are additionally populated with strain
        GEM reactions so that the models can be easily influenced by alterting 
        the values of an allele.
        The change is reverted upon exit when using the model as a context.
        Parameters
        ----------
        processes : Number of cores to parallelize the population optimization
        """
        

        cobra_solution_dict = {}
        if self.base_cobra_model:

            if parallel==True:
                react_set = []
                if fva_rxn_set=="all_reacts":
                    react_set=list([x.id for x in self.base_cobra_model.reactions])
                elif fva_rxn_set=="var_reacts":
                    for x in self.alleles:
                        react_set.extend(x.cobra_reactions.keys())
                    react_set = list(set(react_set))
                self.solution = models_optimize_parallel(self, react_set, fract_opt=fract_opt, save_file_loc=None, fva=fva, processes=processes)
            
            else:
                for strain in tqdm(self.strains):
                    if strain._cobra_model:
                        cobra_solution_dict[strain.id] = strain._cobra_model.optimize()
                    else:
                        raise ValueError('No cobra model in strain: run "update_strains_cobra_model()" to fix')
                self.solution = cobra_solution_dict
        else:
            raise ValueError('No base cobra model found in  {}')
        

    def add_strains(self, strain_list):
        """Add strains to the Cobra4all model object.
        Strains with identifiers identical to a strain already in the
        model are ignored.
        The change is reverted upon exit when using the model as a context.
        Parameters
        ----------
        strain_list : list
            A list of `cobra4all.Strain` objects
        """
        def existing_filter(strain):
            if strain.id in self.strains:
                LOGGER.warning(
                    "Ignoring strain '%s' since it already exists.", strain.id)
                return False
            return True

        # First check whether the reactions exist in the model.
        pruned = DictList(filter(existing_filter, strain_list))

        context = get_context(self)

        # Add reactions. Also take care of genes and metabolites in the loop.
        for strain in pruned:
            strain._model = self
            
            ### Significantly slows down the filter_model_genes and from_matrix functions
            ### because the models are copied then...
            # if self.base_cobra_model:
            #     strain._cobra_model = self.base_cobra_model.copy()
            # else:
            strain._cobra_model = None
            
            # Build a `list()` because the dict will be modified in the loop.
            for allele in list(strain.alleles):
                # TODO: Should we add a copy of the metabolite instead?
                if allele not in self.allele:
                    self.add_alleles(allele)
                # A copy of the metabolite exists in the model, the reaction
                # needs to point to the metabolite in the model.
#                 else:
#                     # FIXME: Modifying 'private' attributes is horrible.
#                     stoichiometry = reaction._metabolites.pop(metabolite)
#                     model_metabolite = self.metabolites.get_by_id(
#                         metabolite.id)
#                     reaction._metabolites[model_metabolite] = stoichiometry
#                     model_metabolite._reaction.add(reaction)
#                     if context:
#                         context(partial(
#                             model_metabolite._reaction.remove, reaction))

#             for gene in list(reaction._genes):
#                 # If the gene is not in the model, add it
#                 if not self.genes.has_id(gene.id):
#                     self.genes += [gene]
#                     gene._model = self

#                     if context:
#                         # Remove the gene later
#                         context(partial(self.genes.__isub__, [gene]))
#                         context(partial(setattr, gene, '_model', None))

#                 # Otherwise, make the gene point to the one in the model
#                 else:
#                     model_gene = self.genes.get_by_id(gene.id)
#                     if model_gene is not gene:
#                         reaction._dissociate_gene(gene)
#                         reaction._associate_gene(model_gene)

        self.strains += pruned

        if context:
            context(partial(self.strains.__isub__, pruned))

#         # from cameo ...
#         self._populate_solver(pruned)
        
    def add_alleles(self, allele_list):
            """Will add a list of alleles to the model object and add new
            constraints accordingly.
            The change is reverted upon exit when using the model as a context.
            Parameters
            ----------
            allele_list : A list of `cobra.core.Allele` objects
            """
            if not hasattr(allele_list, '__iter__'):
                allele_list = [allele_list]
            if len(allele_list) == 0:
                return None

            # First check whether the metabolites exist in the model
            allele_list = [x for x in allele_list
                               if x.id not in self.alleles]

            bad_ids = [m for m in allele_list
                       if not isinstance(m.id, string_types) or len(m.id) < 1]
            if len(bad_ids) != 0:
                raise ValueError('invalid identifiers in {}'.format(repr(bad_ids)))

            for x in allele_list:
                x._model = self
            self.alleles += allele_list

            context = get_context(self)
            if context:
                context(partial(self.alleles.__isub__, allele_list))
                for x in allele_list:
                    # Do we care?
                    context(partial(setattr, x, '_model', None))
                    
                    
    def remove_alleles(self, allele_list, destructive=False):
        """Remove a list of alleles from the the object.
        The change is reverted upon exit when using the model as a context.
        Parameters
        ----------
        allele_list : list
            A list with `cobraSpecies.Allele` objects as elements.
        destructive : bool
            If False then the metabolite is removed from all
            associated reactions.  If True then all associated
            reactions are removed from the Model.
        """
        if not hasattr(allele_list, '__iter__'):
            allele_list = [allele_list]
        # Make sure metabolites exist in model
        allele_list = [x for x in allele_list
                           if x.id in self.alleles]
        for x in allele_list:
            x._model = None

            if not destructive:
                for the_strain in list(x._strain):
                    the_coefficient = the_strain._alleles[x]
                    the_strain.subtract_alleles({x: the_coefficient}) # needs to subtract from list...

            else:
                for x in list(x._reaction):
                    x.remove_from_model()

        self.metabolites -= metabolite_list

        to_remove = [self.solver.constraints[m.id] for m in metabolite_list]
        self.remove_cons_vars(to_remove)

        context = get_context(self)
        if context:
            context(partial(self.metabolites.__iadd__, metabolite_list))
            for x in metabolite_list:
                context(partial(setattr, x, '_model', self))
                
                
    def remove_from_model(self, destructive=False):
        """Removes the association from self.model
        The change is reverted upon exit when using the model as a context.
        Parameters
        ----------
        destructive : bool
            If False then the metabolite is removed from all
            associated reactions.  If True then all associated
            reactions are removed from the Model.
        """
        self._model.remove_metabolites(self, destructive)
                
                
    def copy(self):
        """Provides a partial 'deepcopy' of the Model.  All of the Metabolite,
        Gene, and Reaction objects are created anew but in a faster fashion
        than deepcopy
        """
        new = self.__class__()
        do_not_copy_by_ref = {"alleles", "strains", "base_cobra_model", "notes",
                              "annotation"}
        for attr in self.__dict__:
            if attr not in do_not_copy_by_ref:
                new.__dict__[attr] = self.__dict__[attr]
        new.notes = deepcopy(self.notes)
        new.annotation = deepcopy(self.annotation)

        new.alleles = DictList()
        do_not_copy_by_ref = {"_strains", "_model"}
        for allele in self.alleles:
            new_allele = allele.__class__()
            for attr, value in iteritems(allele.__dict__):
                if attr not in do_not_copy_by_ref:
                    new_allele.__dict__[attr] = copy(
                        value) if attr == "formula" else value
            new_allele._model = new
            new.alleles.append(new_allele)

        new.strains = DictList()
        do_not_copy_by_ref = {"_model", "_alleles", "_base_cobra_model"}
        for strain in self.strains:
            new_strain = strain.__class__()
            for attr, value in iteritems(strain.__dict__):
                if attr not in do_not_copy_by_ref:
                    new_strain.__dict__[attr] = copy(value)
            new_strain._model = new
            new.strains.append(new_strain)
            # update awareness
            for allele, stoic in iteritems(strain._alleles):
                new_allele = new.alleles.get_by_id(allele.id)
                new_strain._alleles[new_allele] = stoic
                new_allele._strain.add(new_strain)
        # it doesn't make sense to retain the context of a copied model so
        # assign a new empty context
        new._contexts = list()
        
    def __enter__(self):
        """Record all future changes to the model, undoing them when a call to
        __exit__ is received"""

        # Create a new context and add it to the stack
        try:
            self._contexts.append(HistoryManager())
        except AttributeError:
            self._contexts = [HistoryManager()]

        return self
        
    def __exit__(self, type, value, traceback):
        """Pop the top context manager and trigger the undo functions"""
        context = self._contexts.pop()
        context.reset()
        
    def _repr_html_(self):
        return """
        <table>
            <tr>
                <td><strong>Name</strong></td>
                <td>{name}</td>
            </tr><tr>
                <td><strong>Memory address</strong></td>
                <td>{address}</td>
            </tr><tr>
                <td><strong>Number of alleles</strong></td>
                <td>{num_alleles}</td>
            </tr><tr>
                <td><strong>Number of strains</strong></td>
                <td>{num_strains}</td>
            </tr><tr>
                <td><strong>base cobra model</strong></td>
                <td>{base_cobra_mod}</td>
            </tr>
          </table>""".format(
            name=self.id,
            address='0x0%x' % id(self),
            num_alleles=len(self.alleles),
            num_strains=len(self.strains),
            base_cobra_mod=self.base_cobra_model)
    
    
    
    
class Variant(Object):
    """Variant is a class for holding information regarding an Allele
    Parameters
    ----------
    id : string
       An identifier for the allele
    name : string
       A human readable name.
    """

    def __init__(self, id=None, name=None):
        Object.__init__(self, id, name)
        self._model = None
        # references to strains that depend on this variant
        self._strain = set()
        self._cobra_gene = None

    @property
    def strains(self):
        return frozenset(self._strain)
    
    @property
    def cobra_gene(self):
        return self._cobra_gene
    
    @cobra_gene.setter
    def cobra_gene(self, gene_id):
        self._cobra_gene = gene_id

    def __getstate__(self):
        """Remove the references to container reactions when serializing to
        avoid problems associated with recursion.
        """
        state = Object.__getstate__(self)
        state['_strain'] = set()
        return state

    def copy(self):
        """When copying a strain, it is necessary to deepcopy the
        components so the list references aren't carried over.
        Additionally, a copy of a strain is no longer in a Cobra4all.Model.
        This should be fixed with self.__deepcopy__ if possible
        """
        return deepcopy(self)

    @property
    def model(self):
        return(self._model)

    
class Allele(Variant):
    """Allele is a class for holding information regarding
    an allele in a CobraScape Species.Strain object. This is required
    in addition to Variant class because the Variant class does not
    tie alleles to strains.
    
    Parameters
    ----------
    id : string
        The identifier to associate with this allele
    name : string
        A human readable name for the allele
    gene:
        The gene identifier of the genetic variant (same as in GEM)
    strains : string
        Strains that contain the allele
    reaction: float
        The reaction(s) catalyzed by the allele 
        
    ----- Game4all specific -----
    actions: float
        The upper flux bound
    """
    
    def __init__(self, id=None, name=''):
        Variant.__init__(self, id, name)
        # The cobra4all.Strains that are used to catalyze the reaction
        self._strain = set()
        # The cobra.Gene corresponding to this allele
        self._cobra_gene = None
        self._cobra_reactions = {}
        # The cobra.Reaction(s) corresponding to this allele
        # self._reaction = set()
    
    @property
    def strains(self):
        return self._strain.copy()
    
    @property
    def cobra_gene(self):
        return self._cobra_gene
    
    @cobra_gene.setter
    def cobra_gene(self, gene_id):
        self._cobra_gene = gene_id
        # update_forward_and_reverse_bounds(self, 'lower')
    
    @property
    def cobra_reactions(self):
        return self._cobra_reactions
    
#     @cobra_reactions.setter
#     def cobra_reactions(self, allele_reacts):
#         self._cobra_reactions = self._cobra_reactions.extend(DictList(allele_reacts))
#         # update_forward_and_reverse_bounds(self, 'lower')

# from cobra.core.object import Object

class Strain(Object):
    """Strain is a class for holding information regarding
    a strain in a pan-genome allele matrix.
    
    Parameters
    ----------
    id : string
        The identifier to associate with this allele
    name : string
        A human readable name for the allele
    alleles : string
        Alleles that the strain contains
    cobra_model: Cobrapy model object
        A unique genome-scale model corresponding to the Strain
    phenotypic: dictionary
        A mapping of strain to observed phenotypic trait {"Trait_1": 0, 
        "Trait_2": 1,... }, where 1 represents the measured presence of 
        the trait and 0 represents the measured absence of the trait.
    """
    def __init__(self, id=None, name=''):
        Object.__init__(self, id, name)
        # The cobra4all.Alleles that are used to catalyze the reaction
        self._alleles = []
        # self.model is None or refers to the cobra4all.Model that contains self
        self._allele_genes = []
        self._model = None
        self._cobra_model = None
        # self._phenotypes = None
        
    @property
    def alleles(self):
        return self._alleles #.copy()
        
    @property
    def allele_genes(self):
        return self._allele_genes #.copy()
    
    @property
    def model(self):
        return self._model #.copy()
    
    @property
    def cobra_model(self):
        return self._cobra_model
        # return self._model.base_cobra_model.copy()
        
    # @property
    # def phenotypes(self):
    #     return self._phenotypes #.copy()
        
    
    def add_alleles(self, alleles_to_add):
        """Add alleles to the strain.
        If the final coefficient for a metabolite is 0 then it is removed
        from the reaction.
        The change is reverted upon exit when using the model as a context.
        Parameters
        ----------
        alleles_to_add : dict
            Dictionary with allele objects or metabolite identifiers as
            keys and 0/1 (presence/absence) as values. If keys are strings (name of a
            allele) the strain must already be part of a model and a
            allele with the given name must exist in the model.
        """
        old_coefficients = self.alleles
        new_alleles = []
        _id_to_alleles = dict([(x.id, x) for x in self._alleles])
        # print _id_to_alleles

        for allele, coefficient in iteritems(alleles_to_add):

            # Make sure alleles being added belong to the cobra4all object, or else copy them.
            if isinstance(allele, Allele):
                if ((allele.model is not None) and
                        (allele.model is not self._model)):
                    allele = allele.copy()

            allele_id = str(allele)
            # If an allele already exists in the cobra4all object, then just add them.
            if allele_id in _id_to_alleles:
                strain_allele = _id_to_alleles[met_id]
                if coefficient == 1:
                    # self._alleles[strain_allele] = coefficient
                    self._alleles.append([strain_allele]) # = coefficient
            else:
                # If the strain is in the Cobra4all model, ensure we aren't using
                # a duplicate allele.
                if self._model:
                    try:
                        allele = \
                            self._model.alleles.get_by_id(allele_id)
                    except KeyError as e:
                        if isinstance(allele, Allele):
                            new_alleles.append(allele)
                        else:
                            # do we want to handle creation here?
                            raise e
                elif isinstance(allele, string_types):
                    # if we want to handle creation, this should be changed
                    raise ValueError("Strain '%s' does not belong to a "
                                     "Cobra4all model. Either add the strain to a "
                                     "Cobra4all model or use Allele objects instead "
                                     "of strings as keys."
                                     % self.id)
                # self._alleles[allele] = coefficient
                if coefficient == 1:
                    self._alleles.append(allele)
                    # make the allele aware that it is involved in this strain
                    allele._strain.add(self)
                    
#     def subtract_alleles(self, alleles, combine=True, reversibly=True):
#         """Subtract metabolites from a reaction.
#         That means add the metabolites with -1*coefficient. If the final
#         coefficient for a metabolite is 0 then the metabolite is removed from
#         the reaction.
#         Notes
#         -----
#         * A final coefficient < 0 implies a reactant.
#         * The change is reverted upon exit when using the model as a context.
#         Parameters
#         ----------
#         metabolites : dict
#             Dictionary where the keys are of class Metabolite and the values
#             are the coefficients. These metabolites will be added to the
#             reaction.
#         combine : bool
#             Describes behavior a metabolite already exists in the reaction.
#             True causes the coefficients to be added.
#             False causes the coefficient to be replaced.
#         reversibly : bool
#             Whether to add the change to the context to make the change
#             reversibly or not (primarily intended for internal use).
#         """
#         self.add_alleles({
#             k: -v for k, v in iteritems(metabolites)},
#             combine=combine, reversibly=reversibly)

### ---------------------------------------------------------------
### -------------------- Simulating CobraScape --------------------
### ---------------------------------------------------------------
import numpy as np
from os import listdir
from os.path import isfile, join
import json

def save_json_obj(sim_json, file_name):
    with open(file_name, 'w') as fp:
        json.dump(sim_json, fp)

def load_json_obj(file_name):
    with open(file_name, 'r') as fp:
        data = json.load(fp)
        return data

### CobraScape Functionality
from cobra.flux_analysis.variability import find_blocked_reactions, flux_variability_analysis
from cobra import Reaction
    
def add_metabolite_drain_reactions(mod_, metab_list, prefix_id="MDS"):
    """Take in a model and add metabolite drain reactions for a list of metabolites
    in the model. These metabolite drain reactions will have the identification
    of (MDS)__(metabolite id). (i.e., MDS__atp_c for the atp_c metabolite)
    """
    mod = mod_.copy()
    metab_obj_names=[]
    for metab in metab_list:
        obj_name = prefix_id+"__"+metab
        metab_drain = Reaction(obj_name)
        metab_drain.lower_bound = 0
        metab_drain.upper_bound = 1000.
        metab_drain.add_metabolites({mod.metabolites.get_by_id(metab): -1.0})
        mod.add_reaction(metab_drain)
        metab_obj_names.append(obj_name)
    return mod, metab_obj_names


def clean_base_model(mod_, open_exchange=False, verbose=True):
    ### Removes reactions that are blocked with the constraints set on exchange reaction flux
    mod = mod_.copy()
    before_mod_genes=len(mod.genes)
    blcked_reacts = find_blocked_reactions(mod, reaction_list=None, zero_cutoff=1e-6, open_exchanges=open_exchange)
    mod.remove_reactions(blcked_reacts, remove_orphans=True)
    check_blcked_reacts = find_blocked_reactions(mod, reaction_list=None, zero_cutoff=1e-6, open_exchanges=open_exchange)
    if verbose==True:
        print("# genes=",str(before_mod_genes),"-> removing",len(blcked_reacts),"blocked reactions","-> # genes=",len(mod.genes))
    return mod


def init_fva_constraints(mod_, opt_frac=0.1, pfba_fact=1.5, verbose=True):
    """Initializes the base cobra model using flux variability analysis.
       Important for getting meaningful results from flux sampling the simulations."""
    if verbose==True:
        print("...constraining the base cobra model with FVA +  pfba constraint")
    mod = mod_.copy()
    fva_df = flux_variability_analysis(mod, fraction_of_optimum=opt_frac, pfba_factor=pfba_fact)
    for rxn, row in fva_df.iterrows():
        if abs(row["maximum"] - row["minimum"]) > 1e-09:
            mod.reactions.get_by_id(rxn).lower_bound = row["minimum"]
            mod.reactions.get_by_id(rxn).upper_bound = row["maximum"]
    return mod, fva_df


# def create_strain_specific(self, gapfill=False):
#     mod = mod_.copy()
#     return mod


def create_action_set(number_of_actions=4, add_no_change=True):
    """Takes N number of reactions and returns labels for actions. This convenience
    function is setup such that the actions are split up into N/2 number of "decreasing"
    and N/2 number of "increasing"
    """
    # TO DO:
    # - check if the number of actions is equal.
    split_num = number_of_actions/2
    action_list = []
    for x in range(int(split_num)):
        action_list.append("lb"+"_"+str(x))
        action_list.append("ub"+"_"+str(x))

    if add_no_change == True:
        action_list.append("no_change")
    return action_list


### Older version of that had issues where the actions were at the minimum and maximum instead of inbetween
def rxn_to_constraints_samples(player_list, action_list, samps):
    """ Assign flux values to constraints for each allele-catalyzed reaction using FVA
        Input:
            - list of Allele objects, list of actions, the fva df
        Returns:
            - dictionary of actions to bound changes.
            ex:  {"lb_0": -2.1, "lb_1": -1.5, .... "ub_0": .8, ....}
    """
    allele_rxns_constraint_dict = {}
    for all_player in player_list:
        allele_rxns_constraint_dict[all_player] = {}
        for react in all_player.cobra_reactions.keys():
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


### Corrected version of rxn_to_constraints_samples
def rxn_to_constraints_samples_v2(player_list, action_list, samps):
    """ Assign flux values to constraints for each allele-catalyzed reaction using FVA
        Input:
            - list of Allele objects, list of actions, the fva df
        Returns:
            - dictionary of actions to bound changes.
            ex:  {"lb_0": -2.1, "lb_1": -1.5, .... "ub_0": .8, ....}
    """
    allele_rxns_constraint_dict = {}
    for all_player in player_list:
        allele_rxns_constraint_dict[all_player] = {}
        for react in all_player.cobra_reactions.keys():
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

                gradient_steps = int(len(action_list)/2)
                # min_to_mean_grad = np.arange(min_flux, mean_flux, (mean_flux-min_flux)/gradient_steps)
                # max_to_mean_grad = np.arange(mean_flux, max_flux, (max_flux-mean_flux)/gradient_steps)
                min_to_mean_grad = np.arange(min_flux, mean_flux, (mean_flux-min_flux)/(gradient_steps+1))[-gradient_steps:]
                max_to_mean_grad = np.arange(mean_flux, max_flux, (max_flux-mean_flux)/(gradient_steps+1))[-gradient_steps:]

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


def rxn_to_constraints_fva(player_list, action_list, fva_frame):
    """ Assign flux values to constraints for each allele-catalyzed reaction using FVA
        Input:
            - list of Allele objects, list of actions, the sampled model df, reaction
        Returns:
            - dictionary of actions to bound changes.
            ex:  {"lb_0": -2.1, "lb_1": -1.5, .... "ub_0": .8, ....}
    """
    allele_rxns_constraint_dict = {}
    for all_player in player_list:
        allele_rxns_constraint_dict[all_player] = {}
        for react in all_player.cobra_reactions.keys():
            allele_rxns_constraint_dict[all_player][react] = {}
            max_flux, min_flux = max(fva_frame[react]), min(fva_frame[react])
            mean_flux = np.mean(fva_frame[react])
    
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


def get_gene_players(gene_list, s_model, verbose=True):
    """Takes in list of genes whose alleles will be accounted for in the popFVA computations
    Returns:
        gene_players - list of relevant alleles as cobrascape objectives
        player_react_list - list of relevant reactions encoded by the alleles
        player_metab_list - list of relevant metabolites acted on by the alleles
    """
    allele_reacts_LIST, allele_list = [], []
    AMR_ALLELES_in_model = [x.id for x in s_model.alleles if x.id.split("_")[0] in gene_list]
    for allele in AMR_ALLELES_in_model:
        allele_reacts=s_model.alleles.get_by_id(allele).cobra_reactions.keys()
        allele_reacts_LIST.extend(allele_reacts)
        # if len(allele_reacts)!=0:
        allele_list.append(allele)

    gene_players = list(set([x.split("_")[0] for x in allele_list]))
    players, allele_ids = [], []
    for allele_player in s_model.alleles:
        if allele_player.cobra_gene in gene_players:
            allele_ids.append(allele_player.id)
            players.append(allele_player)
    player_reacts = []
    for allele in players:
        g=allele.id.split('_')[0]
        g_rxns = [x.id for x in s_model.base_cobra_model.genes.get_by_id(g).reactions]
        player_reacts.extend(g_rxns)
    player_react_list = list(set(player_reacts))
    player_metabs = []
    for rxn in player_react_list:
        rxn_metabs = [x.id for x in s_model.base_cobra_model.reactions.get_by_id(rxn).metabolites]
        player_metabs.extend(rxn_metabs)
    player_metab_list = list(set(player_metabs))

    if verbose==True:
        print("\t...filtered gene list= (# genes: %d, # alleles: %d, # reactions: %d, # metabolites: %d)" % (len(gene_players),
            len(allele_ids),len(player_react_list),len(player_metab_list)))
    return players, player_react_list, player_metab_list
        


### 
# ### Load example dataset
#
# `sampling information`
# Default optG good with 100 samples, fails after 100.
#
# - %time s = sample(TB_model, 100, method="achr")
# CPU times: user 26.7 s, sys: 478 ms, total: 27.1 s
# Wall time: 24.2 s
#
# - %time s = sample(TB_model, 100,processes=4)
# CPU times: user 24.6 s, sys: 383 ms, total: 25 s
# Wall time: 22.8 s
#
# - `%time s = sample(TB_model, 1000, method="achr")
# CPU times: user 39.9 s, sys: 330 ms, total: 40.3 s
# Wall time: 37.6 s`
#     - Use "achr" with large sample number!

#
### ---------------------------------------------------------------------------------
### ------- Perform simulations on landscape using either randomized sampling -------
### ---------------------------------------------------------------------------------
from multiprocessing import cpu_count
from pebble import ProcessPool, ThreadPool
from concurrent.futures import TimeoutError


Species_Object_Global=None
Reaction_List_Global=None
Fraction_Opt_Global=None
def species_init_Objective(mod, react_set, fract_opt):
    global Species_Object_Global
    global Reaction_List_Global
    global Fraction_Opt_Global
    Species_Object_Global = mod
    Reaction_List_Global = react_set
    Fraction_Opt_Global = fract_opt
    del mod
    del react_set
    del fract_opt


def models_optimize_objective(x):
    result_return = (x, Species_Object_Global.strains.get_by_id(x).cobra_model.optimize())
    return result_return


def models_optimize_fva(x):
    result_return = (x, flux_variability_analysis(Species_Object_Global.strains.get_by_id(x).cobra_model, 
                                                  reaction_list=Reaction_List_Global, 
                                                  fraction_of_optimum=Fraction_Opt_Global,
                                                  processes=1).T.to_dict())
                                                  
    return result_return 


def models_optimize_parallel(model, react_set, fract_opt=0.1, save_file_loc=None, fva=False, processes=cpu_count()):
    """ Performs population flux variability analysis (popFVA) in parallel using Pebble
        model: CobraScape Species object
        react_set: list of model reactions to be minimized and maximized.

        if fva=False, the simulation defaults to optimizing the objective function for each strain-GEM. Thus you
        should initialize the strain-GEM objective function prior to running this function with fva=False if you
        want a particular objective to be optimized.
    """
    with ProcessPool(processes, initializer=species_init_Objective, initargs=(model,react_set,fract_opt)) as pool:
        try:
            if fva==True:
                future = pool.map(models_optimize_fva, [x.id for x in model.strains], timeout=100)
                future_iterable = future.result()
                pheno = list(future_iterable)
            else:
                future = pool.map(models_optimize_objective, [x.id for x in model.strains], timeout=20)
                future_iterable = future.result()
                pheno = list(future_iterable)
        except TimeoutError as error:
            print("function took longer than %d seconds" % error.args[1])
        except Exception as error:
            print("function raised %s" % error)

        if save_file_loc!=None:
            save_json_obj(dict(pheno), save_samples_dir+"sample_"+str(t)+".json")
            save_json_obj(variant_id_dec_dict, save_samples_dir+"sample_"+str(t)+"_varDecision.json")
        
    pool.close()
    pool.stop()
    pool.join()
    return pheno
    

### ---------------------------------------------------------------------------------
### --- Perform random sampling of game landscape to compare to learned landscape ---
### ---------------------------------------------------------------------------------
def sample_species(model,save_samples_dir,variants,flux_samples, fva_rxn_set="var_reacts", start_samp=None,
                    samples_n=10,fva=False,fva_frac_opt=0.1,action_n=6,add_na_bound=True,processes=cpu_count()):
    """ sample_species generates random models samples consisting of a random allele-constraint map 
        and a corresponding popFVA landscape
    
    Parameters
    ----------
    model : CobraScape Species object 
        The landscape the alleles will play on
    name : string
        A human readable name for the allele
    alleles : string
        Alleles that the strain contains
    model: float
        A unique genome-scale model corresponding to this strain
    """
    ### Create action set for each player and mapping to model changes.
    actions = create_action_set(number_of_actions=action_n, add_no_change=add_na_bound) ### IMPORTANT- THIS IS UPDATED!!
    variant_rxn_action_dict = rxn_to_constraints_samples_v2(variants, actions, flux_samples)
    # variant_rxn_action_dict = rxn_to_constraints_samples(variants, actions, flux_samples)

    ### Get list of reactions used in popFVA. Won't be used if fva = False, since single objective is solved.
    popfva_reacts_set = []
    if fva_rxn_set=="all_reacts":
        popfva_reacts_set=list([x.id for x in model.base_cobra_model.reactions])
    elif fva_rxn_set=="var_reacts":
        for x in variants:
            popfva_reacts_set.extend(x.cobra_reactions.keys())
        popfva_reacts_set = list(set(popfva_reacts_set))
    
    ### initalize variables
    variant_indices = range(len(variants))
    mixed_action = [[1.0/len(actions) for y in actions ] for x in variant_indices]
    
    species_pheno_trajectory = {}
    pheno = None
    onlyfiles = [f for f in listdir(save_samples_dir) if isfile(join(save_samples_dir, f))]
    onlyfiles = [f for f in onlyfiles if f != ".DS_Store"]
    int_list = [int(x.split("_")[1].strip(".json")) for x in onlyfiles]
    if len(int_list)>0:
        start_t = max(int_list)+1
    else:
        start_t = 0

    if start_samp!=None: # For generating samples on another computer so the 2 samplers don't overlap.
        start_t = start_samp

    print("output: sampling",str(samples_n),"points from the", str(model.id), "landscape, start_t=", start_t)

    ALLELE_REACT_DF = pd.DataFrame()
    for allele in variants[:]:
        for rxn_id in allele.cobra_reactions.keys():
            ALLELE_REACT_DF[allele.id+"__"+rxn_id] = model.strain_allele_matrix[allele.id].replace(1.0, rxn_id)
    ALLELE_REACT_DF.replace(0.0, np.nan, inplace=True)
    ALLELE_REACT_DF["RXN_DUPS"] = ALLELE_REACT_DF.apply(lambda x: x.value_counts()[x.value_counts()>1].index.tolist(),axis=1)

    for t in tqdm(range(samples_n)):
        t = t + start_t
        
        ### --- Sample decision from variants's mixed action at start of round
        decision = [np.random.choice(actions, size = 1, p = mixed_action[x])[0] for x in variant_indices]
        variant_decision_dict = {variants[x]: decision[x] for x in variant_indices}
        variant_id_dec_dict = {variants[x].id: decision[x] for x in variant_indices}
        # with model:
        rxn_bnd_tracker={}
        for allele_play, constrnt in variant_decision_dict.items():
            ### if lb constraint is picked, use base model ub for ub constraint
            if constrnt.split("_")[0] == "lb":
                for react, constrnt_actions in variant_rxn_action_dict[allele_play].items():
                    base_bound = model.base_cobra_model.reactions.get_by_id(react).upper_bound
                    for strain_react in allele_play.cobra_reactions[react]:
                        strain_react.upper_bound = base_bound
                        strain_react.lower_bound = constrnt_actions[constrnt]
            ### if ub constraint is picked, use base model lb for lb constraint            
            elif constrnt.split("_")[0] == "ub":
                for react, constrnt_actions in variant_rxn_action_dict[allele_play].items():
                    base_bound = model.base_cobra_model.reactions.get_by_id(react).lower_bound
                    for strain_react in allele_play.cobra_reactions[react]:
                        strain_react.upper_bound = constrnt_actions[constrnt]
                        strain_react.lower_bound = base_bound
                        
            elif constrnt == 'no_change':
                for react, constrnt_actions in variant_rxn_action_dict[allele_play].items():
                    base_lower_bound = model.base_cobra_model.reactions.get_by_id(react).lower_bound
                    base_upper_bound = model.base_cobra_model.reactions.get_by_id(react).upper_bound
                    for strain_react in allele_play.cobra_reactions[react]:
                        strain_react.upper_bound = base_upper_bound
                        strain_react.lower_bound = base_lower_bound


        ### Take care of duplicate allelic effects for shared reactions
        for strn, allele_row in ALLELE_REACT_DF.iterrows():
            rxn_dups=allele_row["RXN_DUPS"]
            if len(rxn_dups)>0:
                str_obj = model.strains.get_by_id(strn)
                for rxn in rxn_dups:
                    bnd_dict = {"lb":[], "no":[], "ub":[]}
                    alleles={model.alleles.get_by_id(x.split("__")[0]): variant_id_dec_dict[x.split("__")[0]] for x in allele_row[allele_row==rxn].index.tolist()}
                    constraints=[bnd_dict[y.split("_")[0]].append(variant_rxn_action_dict[x][rxn][y]) for x,y in alleles.items()]
                    if len(bnd_dict["lb"])>0:
                        str_obj.cobra_model.reactions.get_by_id(rxn).lower_bound=np.mean(bnd_dict["lb"])
                    else:
                        str_obj.cobra_model.reactions.get_by_id(rxn).lower_bound=model.base_cobra_model.reactions.get_by_id(rxn).lower_bound
                        
                    if len(bnd_dict["ub"])>0:
                        str_obj.cobra_model.reactions.get_by_id(rxn).upper_bound=np.mean(bnd_dict["ub"])
                    else:
                        str_obj.cobra_model.reactions.get_by_id(rxn).upper_bound=model.base_cobra_model.reactions.get_by_id(rxn).upper_bound


        # with ProcessPool(processes, initializer=species_init_Objective, initargs=(model,allele_reacts_set,fva_frac_opt,)) as pool:
        with ProcessPool(processes, initializer=species_init_Objective, initargs=(model,popfva_reacts_set,fva_frac_opt)) as pool:
            try:
                if fva==True:
                    future = pool.map(models_optimize_fva, [x.id for x in model.strains], timeout=100)
                    future_iterable = future.result()
                    pheno = list(future_iterable)
                    save_json_obj(dict(pheno), save_samples_dir+"sample_"+str(t)+"_FVA.json")
                    save_json_obj(variant_id_dec_dict, save_samples_dir+"sample_"+str(t)+"_varDecision.json")
                    del pheno
                else:
                    future = pool.map(models_optimize_objective, [x.id for x in model.strains], timeout=20)
                    future_iterable = future.result()
                    pheno = list(future_iterable)
                    save_json_obj(dict(pheno), save_samples_dir+"sample_"+str(t)+".json")
                    del pheno

            except TimeoutError as error:
                print("function took longer than %d seconds" % error.args[1])
                continue
            except Exception as error:
                print("function raised %s" % error)
                break
        
        pool.close()
        pool.stop()
        pool.join()

    return pool



def compute_constrained_species(model,var_dec_map,variants,flux_samples,fva_rxn_set="all_reacts",
                                fva=False,fva_frac_opt=0.1,action_n=6,add_na_bound=True,processes=cpu_count()):
    """sample_species is a function for the un-biased random sampling of allelic-constraint genome-scale models
    
    Parameters
    ----------
    model : CobraScape Species object 
        The landscape the alleles will play on
    var_dec_map : string
        A human readable name for the allele
    rxn_set : one of the following strings --> ["all_reacts", "var_reacts"]
        List of metabolic reaction ids in base cobra model that will be used in FVA. Used for computational speed up.
    name : string
        A human readable name for the allele
    alleles : string
        Alleles that the strain contains
    model: float
        A unique genome-scale model corresponding to this strain
    """
    ### Create action set for each player and mapping to model changes.
    actions = create_action_set(number_of_actions=action_n, add_no_change=add_na_bound)
    variant_rxn_action_dict = rxn_to_constraints_samples(variants, actions, flux_samples) ### IMPORTANT- THIS IS OLD! v2 available

    ### Get list of reactions used in popFVA. Won't be used if fva = False, since single objective is solved.
    popfva_reacts_set = []
    if fva_rxn_set=="all_reacts":
        popfva_reacts_set=list([x.id for x in model.base_cobra_model.reactions])
    elif fva_rxn_set=="var_reacts":
        for x in variants:
            popfva_reacts_set.extend(x.cobra_reactions.keys())
        popfva_reacts_set = list(set(popfva_reacts_set))
        
    ### initalize variables
    variant_indices = range(len(variants))
    species_pheno_trajectory = {}
    
    pheno = None
    all_pheno_list = []

    ALLELE_REACT_DF = pd.DataFrame()
    for allele in variants[:]:
        for rxn_id in allele.cobra_reactions.keys():
            ALLELE_REACT_DF[allele.id+"__"+rxn_id] = model.strain_allele_matrix[allele.id].replace(1.0, rxn_id)
    ALLELE_REACT_DF.replace(0.0, np.nan, inplace=True)
    ALLELE_REACT_DF["RXN_DUPS"] = ALLELE_REACT_DF.apply(lambda x: x.value_counts()[x.value_counts()>1].index.tolist(),axis=1)
        
    ### --- Sample decision from variants's mixed action at start of round
    variant_decision_dict = {variants[x]: var_dec_map[variants[x].id] for x in variant_indices}

    for allele_play, constrnt in variant_decision_dict.items():
        ### if lb constraint is picked, use base model ub for ub constraint
        if constrnt.split("_")[0] == "lb":
            for react, constrnt_actions in variant_rxn_action_dict[allele_play].items():
                base_bound = model.base_cobra_model.reactions.get_by_id(react).upper_bound
                for strain_react in allele_play.cobra_reactions[react]:
                    strain_react.upper_bound = base_bound
                    strain_react.lower_bound = constrnt_actions[constrnt]
        ### if ub constraint is picked, use base model lb for lb constraint
        elif constrnt.split("_")[0] == "ub":
            for react, constrnt_actions in variant_rxn_action_dict[allele_play].items():
                base_bound = model.base_cobra_model.reactions.get_by_id(react).lower_bound
                for strain_react in allele_play.cobra_reactions[react]:
                    strain_react.upper_bound = constrnt_actions[constrnt]
                    strain_react.lower_bound = base_bound
                    
        elif constrnt == 'no_change':
            for react, constrnt_actions in variant_rxn_action_dict[allele_play].items():
                base_lower_bound = model.base_cobra_model.reactions.get_by_id(react).lower_bound
                base_upper_bound = model.base_cobra_model.reactions.get_by_id(react).upper_bound
                for strain_react in allele_play.cobra_reactions[react]:
                    strain_react.upper_bound = base_upper_bound
                    strain_react.lower_bound = base_lower_bound

    ### Take care of duplicate allelic effects for shared reactions
    for strn, allele_row in ALLELE_REACT_DF.iterrows():
        rxn_dups=allele_row["RXN_DUPS"]
        if len(rxn_dups)>0:
            str_obj = model.strains.get_by_id(strn)
            for rxn in rxn_dups:
                bnd_dict = {"lb":[], "no":[], "ub":[]}
                alleles={model.alleles.get_by_id(x.split("__")[0]): var_dec_map[x.split("__")[0]] for x in allele_row[allele_row==rxn].index.tolist()}
                constraints=[bnd_dict[y.split("_")[0]].append(variant_rxn_action_dict[x][rxn][y]) for x,y in alleles.items()]
                if len(bnd_dict["lb"])>0:
                    str_obj.cobra_model.reactions.get_by_id(rxn).lower_bound=np.mean(bnd_dict["lb"])
                else:
                    str_obj.cobra_model.reactions.get_by_id(rxn).lower_bound=model.base_cobra_model.reactions.get_by_id(rxn).lower_bound
                    
                if len(bnd_dict["ub"])>0:
                    str_obj.cobra_model.reactions.get_by_id(rxn).upper_bound=np.mean(bnd_dict["ub"])
                else:
                    str_obj.cobra_model.reactions.get_by_id(rxn).upper_bound=model.base_cobra_model.reactions.get_by_id(rxn).upper_bound

    with ProcessPool(processes, initializer=species_init_Objective, initargs=(model,popfva_reacts_set,fva_frac_opt)) as pool:
        try:
            if fva==True:
                future = pool.map(models_optimize_fva, [x.id for x in model.strains], timeout=300)
                future_iterable = future.result()
                pheno = list(future_iterable)
            else:
                future = pool.map(models_optimize_objective, [x.id for x in model.strains], timeout=40)
                future_iterable = future.result()
                pheno = list(future_iterable)

        except TimeoutError as error:
            print("function took longer than %d seconds" % error.args[1])
        except Exception as error:
            print("function raised %s" % error)
    
    pool.close()
    pool.stop()
    pool.join()
    return pheno



def set_single_rxn_objective(species_mod, obj_id, obj_direction="max"):
    """ Takes in a single reaction and direction to optimize and sets this objective
        for each strain-specific GEM.
    """
    for strain in tqdm(species_mod.strains):
        species_mod.strains.get_by_id(strain.id).cobra_model.objective=obj_id
        species_mod.strains.get_by_id(strain.id).cobra_model.direction=obj_direction
    return species_mod


### Function for setting objective using a linear function of popFVA features
def set_linear_popfva_objective(species_mod, r_filt_df, obj_dir='max', obj_name="pop_objective"):
    """ Takes in a 1 column dataframe of (popfva features, objective coefficients)
        and sets the objective function for each strain in the cobrascape species
        object accordingly.

        r_filt_df: pandas Series
            Indices are popFVA features (i.e., rxn1_max, rxn1_min, ...)

        The popFVA features are transformed to basic variables in FBA by replacing rxn_max with 
        rxn forward variable and rxn_min with rxn reverse variable.
    """
    # pca_popsol_list = []
    for strain_obj in tqdm(species_mod.strains):
        strain_model = strain_obj.cobra_model #.copy()
        strain_model.objective = strain_model.problem.Objective(0, direction=obj_dir, sloppy=False, 
            name=obj_name)
        strain_model.solver.update()

        objective_coeff_popfva = {}
        for popfva_feat, r_coef in r_filt_df.iterrows():
            r_coef_value = r_coef.values[0]
            if "_max" in popfva_feat:
                rxn_id= popfva_feat.split("_max")[0]
                rxn_dir = "max"
            elif "_min" in popfva_feat:
                rxn_id = popfva_feat.split("_min")[0]
                rxn_dir = "min"
            # print(popfva_feat,rxn_id,rxn_dir, r_coef_value)

            react_obj = strain_model.reactions.get_by_id(rxn_id) #.flux_expression
            if rxn_dir=="min":
                rxn_opt_var = react_obj.reverse_variable
            elif rxn_dir=="max":
                rxn_opt_var = react_obj.forward_variable
            objective_coeff_popfva.update({rxn_opt_var: r_coef_value})

        strain_model.objective.set_linear_coefficients(objective_coeff_popfva)
        objective_obj = strain_model.objective
        species_mod.strains.get_by_id(strain_obj.id).cobra_model.objective=objective_obj
    return species_mod


def get_popobj_sol_df(popsol_obj):
    """ Takes in a non-popFVA single objective population FVA solution and Returns 4 pandas dataframes :
            (strains, shadow prices), (strains, fluxes), (strains, reduced costs), (strains, objective values)
    """
    pop_fluxes = pd.DataFrame()
    pop_sprices = pd.DataFrame()
    pop_rcosts = pd.DataFrame()
    pop_sol = {}
    for strain_id, sol_obj in popsol_obj:
        sprices = sol_obj.shadow_prices.copy()
        fluxes = sol_obj.fluxes.copy()
        rcosts = sol_obj.reduced_costs.copy()
        sol_val = sol_obj.objective_value
        sprices.name=strain_id
        fluxes.name=strain_id
        rcosts.name=strain_id
        pop_fluxes = pd.concat([pop_fluxes, pd.DataFrame(fluxes)], axis=1)
        pop_sprices = pd.concat([pop_sprices, pd.DataFrame(sprices)], axis=1)
        pop_rcosts = pd.concat([pop_rcosts, pd.DataFrame(rcosts)], axis=1)
        pop_sol.update({strain_id: sol_val})
    pop_sol_df = pd.DataFrame.from_dict(pop_sol,orient="index")
    pop_sol_df.columns = ["sol"]
    return pop_fluxes.T, pop_sprices.T, pop_rcosts.T, pop_sol_df


### Helper functions
def filter_pheno_nan(x_species, y_pheno, pheno_name, verbose=False):
    """Filter X and Y to have same index and drop NANs
    """
    y = y_pheno[pheno_name]
    y.dropna(axis=0, how='all', inplace=True)
    X = x_species
    X = X.reindex(y.index)
    return X, y


def duplicate_columns(frame):
    groups = frame.columns.to_series().groupby(frame.dtypes).groups
    dups = []
    for t, v in groups.items():
        cs = frame[v].columns
        vs = frame[v]
        lcs = len(cs)
        for i in range(lcs):
            iv = vs.iloc[:,i].tolist()
            for j in range(i+1, lcs):
                jv = vs.iloc[:,j].tolist()
                if iv == jv:
                    dups.append(cs[i])
                    break
    return dups
