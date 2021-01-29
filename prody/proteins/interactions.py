# -*- coding: utf-8 -*-

"""This module defines functions for calculating different types of interactions 
in protein sttructure, between proteins or between protein and ligand.
Following interactions are availabe for protein interactions:
        (1) Hydrogen bonds
        (2) Salt Bridges
        (3) Repulsive Ionic Bonding 
        (4) Pi stacking interactions
        (5) Pi-cation interactions
        (6) Hydrophobic interactions

For protein-ligand interactions (3) is replaced by water bridges.
"""

__author__ = 'Karolina Mikulska-Ruminska'
__credits__ = ['James Krieger']
__email__ = ['karolami@pitt.edu', 'KRIEGERJ@pitt.edu']


import numpy as np
from prody import LOGGER, SETTINGS
from prody.atomic import AtomGroup, Atom, Atomic, Selection, Select
from prody.atomic import flags
from prody.utilities import importLA, checkCoords
from prody.dynamics.plotting import showAtomicMatrix
from prody.measure import calcDistance, calcAngle, calcCenter
from prody.measure.contacts import findNeighbors
from prody.proteins import writePDB, parsePDB
from collections import Counter


__all__ = ['calcHydrogenBonds', 'calcChHydrogenBonds', 'calcSaltBridges',
           'calcRepulsiveIonicBonding', 'calcPiStacking', 'calcPiCation',
           'calcHydrophohic', 'calcMetalInteractions',
           'calcProteinInteractions', 'calcInteractionMatrix',
           'calcLigandInteractions', 'showLigandInteractions', 
           'showProteinInteractions_VMD', 'showLigandInteraction_VMD', 
           'addHydrogens', 'showHydrogenBondsMap','showInteractionMap']


def cleanNumbers(listContacts):
    """Provide short list with indices and value of distance"""
    
    shortList = [ [int(str(i[0]).split()[-1].strip(')')), 
                           int(str(i[1]).split()[-1].strip(')')), 
                           str(i[0]).split()[1], 
                           str(i[1]).split()[1], 
                           float(i[2])] for i in listContacts ]    
    
    return shortList


def calcPlane(atoms):
    """Function provide paramaters of a plane for aromatic rings (based on 3 points).
    Used in calcPiStacking()"""
    
    coordinates = atoms.getCoords()
    p1, p2, p3 = coordinates[:3] # 3 points will be enough to obtain the plane
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    x3, y3, z3 = p3    
    vec1 = p3 - p1 # These two vectors are in the plane
    vec2 = p2 - p1
    cp = np.cross(vec1, vec2) # the cross product is a vector normal to the plane
    a, b, c = cp
    d = np.dot(cp, p3) # This evaluates a * x3 + b * y3 + c * z3 which equals d
    
    return a,b,c,d


def calcAngleBetweenPlanes(a1, b1, c1, a2, b2, c2):  
    """Find angle between two planes"""
    import math 
          
    d = ( a1 * a2 + b1 * b2 + c1 * c2 ) 
    eq1 = math.sqrt( a1 * a1 + b1 * b1 + c1 * c1) 
    eq2 = math.sqrt( a2 * a2 + b2 * b2 + c2 * c2) 
    d = d / (eq1 * eq2) 
    AngleBetweenPlanes = math.degrees(math.acos(d)) 
    
    return AngleBetweenPlanes
    

def addHydrogens(pdb):    
    """Function will add hydrogens to the protein and ligand structure using Openbabel.
    
    :arg pdb: PDB file name
    :type pdb: str
    
    Instalation of Openbabel:
    conda install -c conda-forge openbabel    

    Find more information here: https://anaconda.org/conda-forge/openbabel
    Program will create new file in the same directory with '_addH.pdb'."""

    try:
        from openbabel import openbabel
        obconversion = openbabel.OBConversion()
        obconversion.SetInFormat("pdb")
        mol = openbabel.OBMol()
        obconversion.ReadFile(mol, pdb)
        mol.AddHydrogens()
        obconversion.WriteFile(mol, pdb[:-4]+'_addH.pdb')
        LOGGER.info("Hydrogens were added to the structure. Structure is saved in the local directry.")
    except:
        LOGGER.info("Install Openbabel to add hydrogens to the structure or use PDBFixer.")
    
    
def calcHydrogenBonds(atoms, distA=3.0, angle=40, cutoff_dist=20, **kwargs):
    """Compute hydrogen bonds for proteins and other molecules.
    
    :arg atoms: an Atomic object from which residues are selected
    :type atoms: :class:`.Atomic`
    
    :arg distA: non-zero value, maximal distance between donor and acceptor.
    :type distA: int or float, default is 3.0
    
    :arg angle: non-zero value, maximal (180 - D-H-A angle) (donor, hydrogen, acceptor).
    :type distA: int or float, default is 40.
    
    :arg cutoff_dist: non-zero value, interactions will be found between residues that are higher than cutoff_dist.
    :type cutoff_dist: int, default is 20 atoms.

    Structre should contain hydrogens.
    If not they can be added using addHydrogens(pdb_name) function availabe in ProDy after Openbabel instalation.
    `conda install -c conda-forge openbabel`
    
    Note that the angle which is considering is 180-defined angle D-H-A (in a good agreement with VMD)
    Results can be displayed in VMD by using showVMDinteraction() """

    try:
        coords = (atoms._getCoords() if hasattr(atoms, '_getCoords') else
                    atoms.getCoords())
    except AttributeError:
        try:
            checkCoords(coords)
        except TypeError:
            raise TypeError('coords must be an object '
                            'with `getCoords` method')
    try:
        donors = kwargs['donors']
        acceptors = kwargs['acceptors']
    except:
        donors=['N', 'O', 'S', 'F']
        acceptors=['N', 'O', 'S', 'F']
    
    if atoms.hydrogen == None or atoms.hydrogen.numAtoms() < 10:
        LOGGER.info("Provide structure with hydrogens or install Openbabel to add missing hydrogens using addHydrogens(pdb_name) first.")
    
    contacts = findNeighbors(atoms, distA)
    short_contacts = cleanNumbers(contacts)
    pairList = [] # list with Donor-Hydrogen-Acceptor(indices)-distance-Angle
    
    LOGGER.info('Calculating hydrogen bonds.')
    for nr_i,i in enumerate(short_contacts):
        # Removing those close contacts which are between neighbour atoms
        if i[1] - cutoff_dist < i[0] < i[1] + cutoff_dist:
            pass
        
        else:
            if (i[2][0] in donors and i[3][0] in acceptors) or (i[2][0] in acceptors and i[3][0] in donors): # First letter is checked
                listOfHydrogens1 = cleanNumbers(findNeighbors(atoms.hydrogen, 1.4, atoms.select('index '+str(i[0]))))
                listOfHydrogens2 = cleanNumbers(findNeighbors(atoms.hydrogen, 1.4, atoms.select('index '+str(i[1]))))
                AtomsForAngle = ['D','H','A', 'distance','angle']
                
                if not listOfHydrogens1:
                    for j in listOfHydrogens2:
                        AtomsForAngle = [j[1], j[0], i[0], i[-1], calcAngle(atoms.select('index '+str(j[1])), 
                                                                        atoms.select('index '+str(j[0])), 
                                                                        atoms.select('index '+str(i[0])))[0]]                                                                                   
                        pairList.append(AtomsForAngle)            
                
                elif not listOfHydrogens2:
                    for jj in listOfHydrogens1:
                        AtomsForAngle = [jj[1], jj[0], i[1], i[-1], calcAngle(atoms.select('index '+str(jj[1])), 
                                                                          atoms.select('index '+str(jj[0])), 
                                                                          atoms.select('index '+str(i[1])))[0]]
                        pairList.append(AtomsForAngle)            
       
                else:            
                    for j in listOfHydrogens2:
                        AtomsForAngle = [j[1], j[0], i[0], i[-1], calcAngle(atoms.select('index '+str(j[1])), 
                                                                            atoms.select('index '+str(j[0])), 
                                                                            atoms.select('index '+str(i[0])))[0]]                                                                                   
                        pairList.append(AtomsForAngle)
    
                    
                    for jj in listOfHydrogens1:
                        AtomsForAngle = [jj[1], jj[0], i[1], i[-1], calcAngle(atoms.select('index '+str(jj[1])), 
                                                                              atoms.select('index '+str(jj[0])), 
                                                                              atoms.select('index '+str(i[1])))[0]]
                        pairList.append(AtomsForAngle)
    
    HBs_list = []
    for k in pairList:
        if 180-angle < float(k[-1]) < 180 and float(k[-2]) < distA:
            aa_donor = atoms.getResnames()[k[0]]+str(atoms.getResnums()[k[0]])
            aa_donor_atom = atoms.getNames()[k[0]]+'_'+str(k[0])
            aa_donor_chain = atoms.getChids()[k[0]]
            aa_acceptor = atoms.getResnames()[k[2]]+str(atoms.getResnums()[k[2]])
            aa_acceptor_atom = atoms.getNames()[k[2]]+'_'+str(k[2])
            aa_acceptor_chain = atoms.getChids()[k[2]]
            
            HBs_list.append([aa_donor, aa_donor_atom, aa_donor_chain, aa_acceptor, aa_acceptor_atom, 
                             aa_acceptor_chain, float(k[-2]), float(k[-1])])
    
    HBs_list = sorted(HBs_list, key=lambda x : x[-2])
    LOGGER.info("Number of detected hydrogen bonds: {0}.".format(len(HBs_list)))
                
    return HBs_list   
    
    
def calcChHydrogenBonds(atoms, distA=3.0, angle=40, cutoff_dist=20, **kwargs):
    """ Compute hydrogen bonds between different chains.
    See more details in calcHydrogenBonds().
    
    :arg atoms: an Atomic object from which residues are selected
    :type atoms: :class:`.Atomic`
    
    :arg distA: non-zero value, maximal distance between donor and acceptor.
    :type distA: int or float, default is 3.0.
    
    :arg angle: non-zero value, D-H-A angle (donor, hydrogen, acceptor).
    :type distA: int or float, default is 40.
    
    :arg cutoff_dist: non-zero value, interactions will be found between residues that are higher than cutoff_dist.
    :type cutoff_dist: int, default is 20 atoms.

    Structre should contain hydrogens.
    If not they can be added using addHydrogens(pdb_name) function availabe in ProDy after Openbabel instalation.
    `conda install -c conda-forge openbabel`
    
    Note that the angle which is considering is 180-defined angle D-H-A (in a good agreement with VMD)
    Results can be displayed in VMD. """

    try:
        coords = (atoms._getCoords() if hasattr(atoms, '_getCoords') else
                    atoms.getCoords())
    except AttributeError:
        try:
            checkCoords(coords)
        except TypeError:
            raise TypeError('coords must be an object '
                            'with `getCoords` method')

    if len(np.unique(atoms.getChids())) > 1:
        HBS_calculations = calcHydrogenBonds(atoms, **kwargs)
    
        ChainsHBs = [ i for i in HBS_calculations if str(i[2]) != str(i[5]) ]
        if not ChainsHBs:
            ligand_name = list(set(atoms.select('all not protein and not ion').getResnames()))[0]
            ChainsHBs = [ ii for ii in HBS_calculations if ii[0][:3] == ligand_name or ii[3][:3] == ligand_name ]

        return ChainsHBs 
        

def calcSaltBridges(atoms, distA=4.5):
    """Searching for salt bridges in protein structure.
    Histidine is not considered as a charge residue
    
    :arg atoms: an Atomic object from which residues are selected
    :type atoms: :class:`.Atomic`
    
    :arg distA: non-zero value, maximal distance between center of masses 
        of N and O atoms of negatively and positevely charged residues.
    :type distA: int or float, default is 4.5.
    
    Results can be displayed in VMD."""

    try:
        coords = (atoms._getCoords() if hasattr(atoms, '_getCoords') else
                    atoms.getCoords())
    except AttributeError:
        try:
            checkCoords(coords)
        except TypeError:
            raise TypeError('coords must be an object '
                            'with `getCoords` method')

    atoms_KRED = atoms.select('protein and resname ASP GLU LYS ARG and not backbone and not name OXT NE "C.*" and noh')
    charged_residues = list(set(zip(atoms_KRED.getResnums(), atoms_KRED.getChids())))
    
    LOGGER.info('Calculating salt bridges.')
    SaltBridges_list = []
    for i in charged_residues:
        sele1 = atoms_KRED.select('resid '+str(i[0])+' and chain '+i[1])
        try:
            sele1_center = calcCenter(sele1.getCoords())
            sele2 = atoms_KRED.select('same residue as exwithin '+str(distA)+' of center', center=sele1_center)
        except:
            sele1_center = sele1.getCoords()
            sele2 = atoms_KRED.select('same residue as exwithin '+str(distA)+' of center', center=sele1.getCoords())            
 
        if sele1 != None and sele2 != None:
            for ii in np.unique(sele2.getResnums()):                
                sele2_single = sele2.select('resid '+str(ii))
                try:
                    distance = calcDistance(sele1_center,calcCenter(sele2_single.getCoords()))
                except: 
                    distance = calcDistance(sele1_center,sele2_single.getCoords())
                
                if distance < distA and sele1.getNames()[0][0] != sele2_single.getNames()[0][0]:
                    SaltBridges_list.append([sele1.getResnames()[0]+str(sele1.getResnums()[0]), sele1.getNames()[0]+'_'+'_'.join(map(str,sele1.getIndices())), sele1.getChids()[0],
                                                  sele2_single.getResnames()[0]+str(sele2_single.getResnums()[0]), sele2_single.getNames()[0]+'_'+'_'.join(map(str,sele2_single.getIndices())), 
                                                  sele2_single.getChids()[0], round(distance,3)])
    
    SaltBridges_list = sorted(SaltBridges_list, key=lambda x : x[-1])
    [ SaltBridges_list.remove(j) for i in SaltBridges_list for j in SaltBridges_list if Counter(i) == Counter(j) ]

    LOGGER.info("Number of detected salt bridges: {0}.".format(len(SaltBridges_list)))        

    return SaltBridges_list
    

def calcRepulsiveIonicBonding(atoms, distA=4.5):
    """Searching for repulsive ionic bondning in protein structure
    i.e. between positive-positive or negative-negative residues.
    Histidine is not considered as a charge residue
    
    :arg atoms: an Atomic object from which residues are selected
    :type atoms: :class:`.Atomic`
    
    :arg distA: non-zero value, maximal distance between center of masses 
            between N-N or O-O atoms of residues.
    :type distA: int or float, default is 4.5.
    
    Results can be displayed in VMD."""

    try:
        coords = (atoms._getCoords() if hasattr(atoms, '_getCoords') else
                    atoms.getCoords())
    except AttributeError:
        try:
            checkCoords(coords)
        except TypeError:
            raise TypeError('coords must be an object '
                            'with `getCoords` method')

    atoms_KRED = atoms.select('protein and resname ASP GLU LYS ARG and not backbone and not name OXT NE "C.*" and noh')
    charged_residues = list(set(zip(atoms_KRED.getResnums(), atoms_KRED.getChids())))
    
    LOGGER.info('Calculating repulsive ionic bonidng.')
    RepulsiveIonicBonding_list = []
    for i in charged_residues:
        sele1 = atoms_KRED.select('resid '+str(i[0])+' and chain '+i[1])
        try:
            sele1_center = calcCenter(sele1.getCoords())
            sele2 = atoms_KRED.select('same residue as exwithin '+str(distA)+' of center', center=sele1_center)
        except:
            sele1_center = sele1.getCoords()
            sele2 = atoms_KRED.select('same residue as exwithin '+str(distA)+' of center', center=sele1.getCoords())            
 
        if sele1 != None and sele2 != None:
            for ii in np.unique(sele2.getResnums()):                
                sele2_single = sele2.select('resid '+str(ii))
                try:
                    distance = calcDistance(sele1_center,calcCenter(sele2_single.getCoords()))
                except: 
                    distance = calcDistance(sele1_center,sele2_single.getCoords())
                
                if distance < distA and sele1.getNames()[0][0] == sele2_single.getNames()[0][0] and sele1.getResnames()[0] != sele2_single.getResnames()[0]:
                    RepulsiveIonicBonding_list.append([sele1.getResnames()[0]+str(sele1.getResnums()[0]), sele1.getNames()[0]+'_'+'_'.join(map(str,sele1.getIndices())), sele1.getChids()[0],
                                                  sele2_single.getResnames()[0]+str(sele2_single.getResnums()[0]), sele2_single.getNames()[0]+'_'+'_'.join(map(str,sele2_single.getIndices())), 
                                                  sele2_single.getChids()[0], round(distance,3)])
    
    [ RepulsiveIonicBonding_list.remove(j) for i in RepulsiveIonicBonding_list for j in RepulsiveIonicBonding_list if Counter(i) == Counter(j) ]
    RepulsiveIonicBonding_list = sorted(RepulsiveIonicBonding_list, key=lambda x : x[-1])
    
    LOGGER.info("Number of detected Repulsive Ionic Bonding interactions: {0}.".format(len(RepulsiveIonicBonding_list)))
    
    return RepulsiveIonicBonding_list


def calcPiStacking(atoms, distA=5.0, angle_min=0, angle_max=360, **kwargs):
    """Searching for π–π stacking interactions (between aromatic rings).
    
    :arg atoms: an Atomic object from which residues are selected
    :type atoms: :class:`.Atomic`
    
    :arg distA: non-zero value, maximal distance between center of masses of residues aromatic rings.
    :type distA: int or float, default is 5.
    
    :arg angle_min: minimal angle between aromatic rings.
    :type angle_min: int, default is 0.

    :arg angle_max: maximal angle between rings.
    :type angle_max: int, default is 360.
    
    Results can be displayed in VMD.
    By default three residues are included TRP, PHE, TYR and HIS.
    Additional selection can be added: 
        >> calcPiStacking(atoms, 'HSE'='noh and not backbone and not name CB')
        or
        >> kwargs = {"HSE": "noh and not backbone and not name CB", "HSD": "noh and not backbone and not name CB"}
        >> calcPiStacking(atoms,**kwargs)
    Predictions for proteins only. 
    To compute protein-ligand interactions use calcLigandInteractions() or define **kwargs"""

    try:
        coords = (atoms._getCoords() if hasattr(atoms, '_getCoords') else
                    atoms.getCoords())
    except AttributeError:
        try:
            checkCoords(coords)
        except TypeError:
            raise TypeError('coords must be an object '
                            'with `getCoords` method')

    aromatic_dic = {'TRP':'noh and not backbone and not name CB NE1 CD1 CG',
                'PHE':'noh and not backbone and not name CB',
                'TYR':'noh and not backbone and not name CB and not name OH',
                'HIS':'noh and not backbone and not name CB'}
    
    for key, value in kwargs.items():
        aromatic_dic[key] = value
    
    atoms_cylic = atoms.select('resname TRP PHE TYR HIS')
    aromatic_resids = list(set(zip(atoms_cylic.getResnums(), atoms_cylic.getChids())))

    LOGGER.info('Calculating Pi stacking interactions.')
    PiStack_calculations = []
    for i in aromatic_resids:
        for j in aromatic_resids:
            if i != j: 
                sele1_name = atoms.select('resid '+str(i[0])+' and chain '+i[1]+' and name CA').getResnames()
                sele1 = atoms.select('resid '+str(i[0])+' and chain '+i[1]+' and '+aromatic_dic[sele1_name[0]])
                
                sele2_name = atoms.select('resid '+str(j[0])+' and chain '+j[1]+' and name CA').getResnames()
                sele2 = atoms.select('resid '+str(j[0])+' and chain '+j[1]+' and '+aromatic_dic[sele2_name[0]])
                
                if sele1 != None and sele2 != None:
                    a1, b1, c1, a2, b2, c2 = calcPlane(sele1)[:3]+calcPlane(sele2)[:3]
                    RingRing_angle = calcAngleBetweenPlanes(a1, b1, c1, a2, b2, c2) # plane is computed based on 3 points of rings           
                    RingRing_distance = calcDistance(calcCenter(sele1.getCoords()),calcCenter(sele2.getCoords()))
                    if RingRing_distance < distA and angle_min < RingRing_angle < angle_max:
                        PiStack_calculations.append([sele1_name[0]+str(sele1.getResnums()[0]), '_'.join(map(str,sele1.getIndices())), sele1.getChids()[0],
                                                     sele2_name[0]+str(sele2.getResnums()[0]), '_'.join(map(str,sele2.getIndices())), sele2.getChids()[0],
                                                     round(RingRing_distance,3), round(RingRing_angle,3)])
    
    PiStack_calculations = sorted(PiStack_calculations, key=lambda x : x[-2])   
    LOGGER.info("Number of detected Pi stacking interactions: {0}.".format(len(PiStack_calculations)))
    
    return PiStack_calculations


def calcPiCation(atoms, distA=5.0, extraSele=None, **kwargs):
    """Searching for cation-Pi interaction i.e. between aromatic ring and positively charged residue (ARG and LYS)
    
    :arg atoms: an Atomic object from which residues are selected
    :type atoms: :class:`.Atomic`
    
    :arg distA: non-zero value, maximal distance between center of masses of aromatic ring and positively charge group.
    :type distA: int or float, default is 5.

    By default three residues are included TRP, PHE, TYR and HIS.
    Additional selection can be added: 
        >> calcPiCation(atoms, 'HSE'='noh and not backbone and not name CB')
        or
        >> kwargs = {"HSE": "noh and not backbone and not name CB", "HSD": "noh and not backbone and not name CB"}
        >> calcPiCation(atoms,**kwargs)
    Results can be displayed in VMD.
    Predictions for proteins only. To compute protein-ligand interactions use calcLigandInteractions() or define **kwargs"""

    try:
        coords = (atoms._getCoords() if hasattr(atoms, '_getCoords') else
                    atoms.getCoords())
    except AttributeError:
        try:
            checkCoords(coords)
        except TypeError:
            raise TypeError('coords must be an object '
                            'with `getCoords` method')
    
    aromatic_dic = {'TRP':'noh and not backbone and not name CB NE1 CD1 CG',
                'PHE':'noh and not backbone and not name CB',
                'TYR':'noh and not backbone and not name CB and not name OH',
                'HIS':'noh and not backbone and not name CB'}
        
    for key, value in kwargs.items():
        aromatic_dic[key] = value
        
    atoms_cylic = atoms.select('resname TRP PHE TYR HIS')
    aromatic_resids = list(set(zip(atoms_cylic.getResnums(), atoms_cylic.getChids())))

    PiCation_calculations = []
    LOGGER.info('Calculating cation-Pi interactions.')
    
    for i in aromatic_resids:
        sele1_name = atoms.select('resid '+str(i[0])+' and chain '+i[1]+' and name CA').getResnames()
        
        try:
            sele1 = atoms.select('resid '+str(i[0])+' and chain '+i[1]+' and '+aromatic_dic[sele1_name[0]])
            sele2 = atoms.select('(same residue as exwithin '+str(distA)+' of center) and resname ARG LYS and noh and not backbone and not name NE "C.*"', 
                               center=calcCenter(sele1.getCoords()))
        except:
            LOGGER.info("Missing atoms from the side chains of the structure. Use PDBFixer.")
        if sele1 != None and sele2 != None:
            for ii in np.unique(sele2.getResnums()):
                sele2_single = sele2.select('resid '+str(ii))
                try:
                    RingCation_distance = calcDistance(calcCenter(sele1.getCoords()),calcCenter(sele2_single.getCoords()))
                except: 
                    RingCation_distance = calcDistance(calcCenter(sele1.getCoords()),sele2_single.getCoords())
                
                if RingCation_distance < distA:
                    PiCation_calculations.append([sele1_name[0]+str(sele1.getResnums()[0]), '_'.join(map(str,sele1.getIndices())), sele1.getChids()[0],
                                                  sele2_single.getResnames()[0]+str(sele2_single.getResnums()[0]), sele2_single.getNames()[0]+'_'+'_'.join(map(str,sele2_single.getIndices())), 
                                                  sele2_single.getChids()[0], round(RingCation_distance,3)])
    
    PiCation_calculations = sorted(PiCation_calculations, key=lambda x : x[-1]) 
    LOGGER.info("Number of detected cation-pi interactions: {0}.".format(len(PiCation_calculations)))

    return PiCation_calculations


def calcHydrophohic(atoms, distA=4.5, **kwargs): 
    """Prediction of hydrophobic interactions between hydrophobic residues (ALA, ILE, LEU, MET, PHE, TRP, VAL).
    
    :arg atoms: an Atomic object from which residues are selected
    :type atoms: :class:`.Atomic`
    
    :arg distA: non-zero value, maximal distance between atoms of hydrophobic residues.
    :type distA: int or float, default is 4.5.
    
    Additional selection can be added as shown below (with selection that includes only hydrophobic part): 
        >> calcHydrophohic(atoms, 'XLE'='noh and not backbone')
    Predictions for proteins only. To compute protein-ligand interactions use calcLigandInteractions() or define **kwargs
    Results can be displayed in VMD by using showVMDinteraction() 
    
    Note that interactions between aromatic residues are omitted becasue they are provided by calcPiStacking().    
    Results can be displayed in VMD."""

    try:
        coords = (atoms._getCoords() if hasattr(atoms, '_getCoords') else
                    atoms.getCoords())
    except AttributeError:
        try:
            checkCoords(coords)
        except TypeError:
            raise TypeError('coords must be an object '
                            'with `getCoords` method')

    Hydrophobic_list = []  
    atoms_hydrophobic = atoms.select('resname ALA VAL ILE MET LEU PHE TYR TRP')
    hydrophobic_resids = list(set(zip(atoms_hydrophobic.getResnums(), atoms_hydrophobic.getChids())))
    
    aromatic_nr = list(set(zip(atoms.aromatic.getResnums(),atoms.aromatic.getChids())))   
    aromatic = list(set(zip(atoms.aromatic.getResnames())))
    
    hydrophobic_dic = {'ALA': 'noh and not backbone', 'VAL': 'noh and not (backbone or name CB)',
    'ILE': 'noh and not (backbone or name CB)', 'LEU': 'noh and not (backbone or name CB)',
    'MET': 'noh and not (backbone or name CB)', 'PHE': 'noh and not (backbone or name CB)',
    'TYR': 'noh and not (backbone or name CB)', 'TRP': 'noh and not (backbone or name CB)'}

    for key, value in kwargs.items():
        hydrophobic_dic[key] = value
    
    LOGGER.info('Calculating hydrophobic interactions.')
    Hydrophobic_calculations = []
    for i in hydrophobic_resids:
        try:
            sele1_name = atoms.select('resid '+str(i[0])+' and chain '+i[1]+' and name CA').getResnames()
            sele1 = atoms.select('resid '+str(i[0])+' and '+' chain '+i[1]+' and '+hydrophobic_dic[sele1_name[0]]) 
            sele1_nr = sele1.getResnums()[0]  
            sele2 = atoms.select('(same residue as exwithin '+str(distA)+' of (resid '+str(sele1_nr)+' and chain '+i[1]+' and resname '+sele1_name[0]+
                               ')) and ('+' or '.join([ '(resname '+item[0]+' and '+item[1]+')' for item in hydrophobic_dic.items() ])+')')

        except:
            LOGGER.info("Missing atoms from the side chains of the structure. Use PDBFixer.")
            sele1 = None
            sele2 = None
        
        if sele2 != None:
            sele2_nr = list(set(zip(sele2.getResnums(), sele2.getChids())))

            if sele1_name[0] in aromatic:
                sele2_filter = sele2.select('all and not (resname TYR PHE TRP or resid '+str(i)+')')
                if sele2_filter != None:
                    listOfAtomToCompare = cleanNumbers(findNeighbors(sele1, distA, sele2_filter))
                
            elif sele1_name[0] not in aromatic and i in sele2_nr:
                sele2_filter = sele2.select(sele2.select('all and not (resid '+str(i[0])+' and chain '+i[1]+')'))
                if sele2_filter != None:
                    listOfAtomToCompare = cleanNumbers(findNeighbors(sele1, distA, sele2_filter))
            else:
                listOfAtomToCompare = cleanNumbers(findNeighbors(sele1, distA, sele2))
                                                           
            if listOfAtomToCompare != []:
                listOfAtomToCompare = sorted(listOfAtomToCompare, key=lambda x : x[-1])
                minDistancePair = listOfAtomToCompare[0]
                if minDistancePair[-1] < distA:
                    sele1_new = atoms.select('index '+str(minDistancePair[0])+' and name '+str(minDistancePair[2]))
                    sele2_new = atoms.select('index '+str(minDistancePair[1])+' and name '+str(minDistancePair[3]))
                    Hydrophobic_calculations.append([sele1_new.getResnames()[0]+str(sele1_new.getResnums()[0]), 
                                                             minDistancePair[2]+'_'+str(minDistancePair[0]), sele1_new.getChids()[0],
                                                             sele2_new.getResnames()[0]+str(sele2_new.getResnums()[0]), 
                                                             minDistancePair[3]+'_'+str(minDistancePair[1]), sele2_new.getChids()[0],
                                                             round(minDistancePair[-1],3)]) 
                    
    Hydrophobic_calculations = sorted(Hydrophobic_calculations, key=lambda x : x[-1])
    LOGGER.info("Number of detected hydrophobic interactions: {0}.".format(len(Hydrophobic_calculations)))
    
    return Hydrophobic_calculations


def calcMetalInteractions(atoms, distA=3.0, extraIons=['FE'], excluded_ions=['SOD', 'CLA']):
    """Interactions with metal ions (includes water, ligands and other ions).
        
    :arg atoms: an Atomic object from which residues are selected
    :type atoms: :class:`.Atomic`
    
    :arg distA: non-zero value, maximal distance between ion and residue.
    :type distA: int or float, default is 3.0.
    
    :arg extraIons: ions to be included in the analysis.
    :type extraIons: list of str
    
    :arg excluded_ions: ions which should be excluded from the analysis.
    :type excluded_ions: list of str """
    
    try:
        coords = (atoms._getCoords() if hasattr(atoms, '_getCoords') else
                    atoms.getCoords())
    except AttributeError:
        try:
            checkCoords(coords)
        except TypeError:
            raise TypeError('coords must be an object '
                            'with `getCoords` method')
    
    try:
        atoms_ions = atoms.select('ion and not name '+' '.join(excluded_ions)+' or (name '+' '.join(map(str,extraIons))+')')
        MetalResList = []
        MetalRes_calculations = cleanNumbers(findNeighbors(atoms_ions, distA, atoms.select('all and noh')))
        for i in MetalRes_calculations:
            if i[-1] != 0:
                MetalResList.append([atoms.getResnames()[i[0]]+str(atoms.getResnums()[i[0]]), i[2], 
                                 atoms.getResnames()[i[1]]+str(atoms.getResnums()[i[1]]), i[3], i[-1]])

        return MetalResList
        
    except TypeError:
        raise TypeError('An object should contain ions')


def calcProteinInteractions(atoms):
    """Compute all protein interactions (shown below) using default paramaters.
        (1) Hydrogen bonds
        (2) Salt Bridges
        (3) RepulsiveIonicBonding 
        (4) Pi stacking interactions
        (5) Pi-cation interactions
        (6) Hydrophobic interactions
    
    :arg atoms: an Atomic object from which residues are selected
    :type atoms: :class:`.Atomic`"""

    try:
        coords = (atoms._getCoords() if hasattr(atoms, '_getCoords') else
                    atoms.getCoords())
    except AttributeError:
        try:
            checkCoords(coords)
        except TypeError:
            raise TypeError('coords must be an object '
                            'with `getCoords` method')

    LOGGER.info('Calculating all interations.') 
    HBs_calculations = calcHydrogenBonds(atoms.protein)               #1 in scoring
    SBs_calculations = calcSaltBridges(atoms.protein)                 #2
    SameChargeResidues = calcRepulsiveIonicBonding(atoms.protein)     #3
    Pi_stacking = calcPiStacking(atoms.protein)                       #4
    Pi_cation = calcPiCation(atoms.protein)                           #5
    Hydroph_calculations = calcHydrophohic(atoms.protein)             #6
    AllInteractions = [HBs_calculations, SBs_calculations, SameChargeResidues, Pi_stacking, Pi_cation, Hydroph_calculations]   
    
    return AllInteractions
    

def calcInteractionMatrix(atoms):
    """Calculate matrix with protein interactions which is scored by as follows:
        (1) Hydrogen bonds +2
        (2) Salt Bridges +3 (Salt bridges might be included in hydrogen bonds)
        (3) Repulsive Ionic Bonding -1 
        (4) Pi stacking interactions +3
        (5) Pi-cation interactions +3
        (6) Hydrophobic interactions +1
       
    :arg atoms: an Atomic object from which residues are selected
    :type atoms: :class:`.Atomic`
    
    Results can be displayed:
        >> plt.spy(calcInteractionMatrix(atoms), origin='lower', markersize=1.5)
        or
        >> showAtomicMatrix(calcInteractionMatrix(atoms), atoms=atoms.select('name CA'), 
            interpolation=None, cmap='seismic'); plt.clim([-3,3])"""
            
    try:
        coords = (atoms._getCoords() if hasattr(atoms, '_getCoords') else
                    atoms.getCoords())
    except AttributeError:
        try:
            checkCoords(coords)
        except TypeError:
            raise TypeError('coords must be an object '
                            'with `getCoords` method')
    
    LOGGER.info('Calculating all interactions')
    AllInteractions = calcProteinInteractions(atoms)
    InteractionsMap = np.zeros([atoms.select('name CA').numAtoms(),atoms.select('name CA').numAtoms()])
    resIDs = list(atoms.select('name CA').getResnums())
    resChIDs = list(atoms.select('name CA').getChids())
    resIDs_with_resChIDs = list(zip(resIDs, resChIDs))
    scoring = [2, 3, -1, 3, 3, 1]  #HBs_calculations, SBs_calculations, SameChargeResidues, Pi_stacking, Pi_cation, Hydroph_calculations
    
    for nr_i,i in enumerate(AllInteractions):
        if i != []:
            for ii in i: 
                m1 = resIDs_with_resChIDs.index((int(ii[0][3:]),ii[2]))
                m2 = resIDs_with_resChIDs.index((int(ii[3][3:]),ii[5]))
                InteractionsMap[m1][m2] = InteractionsMap[m1][m2] + scoring[nr_i]
  
    return InteractionsMap


def calcLigandInteractions(atoms, select='all not (water or protein or ion)', ignore_ligs=['NAG','BMA','MAN']):
    """Provide ligand interactions with other elements of the system including protein, water and ions.
    Results are computed by PLIP program which should be installed in local computer.
    Note that PLIP will not recognize ligand unless it will be HETATM in PDB file.
    
    :arg atoms: an Atomic object from which residues are selected
    :type atoms: :class:`.Atomic`
    
    :arg select: a selection string for residues of interest
            default is 'all not (water or protein or ion)'
    :type select: str
    
    :arg ignore_ligs: List of ligands which will be excluded from the analysis.
    :type ignore_ligs: List of str
    
    To display results as a list of interactions use showLigandInteractions()
    and for visualization in VMD program please use showLigandInteraction_VMD() 
    
    Requirements of usage:
    ## Instalation of Openbabel:
    >> conda install -c conda-forge openbabel    
    ## https://anaconda.org/conda-forge/openbabel
    
    ## Instalation of PLIP
    >> conda install -c conda-forge plip
    ## https://anaconda.org/conda-forge/plip
    # https://github.com/pharmai/plip/blob/master/DOCUMENTATION.md """
    
    try:
        coords = (atoms._getCoords() if hasattr(atoms, '_getCoords') else
                    atoms.getCoords())
    except AttributeError:
        try:
            checkCoords(coords)
        except TypeError:
            raise TypeError('coords must be an object '
                            'with `getCoords` method')
    try:
        from plip.structure.preparation import PDBComplex   
        
        pdb_name = atoms.getTitle()+'_sele.pdb'
        LOGGER.info("Writing PDB file with selection in the local direvtory.")
        writePDB(pdb_name, atoms)

        try:
            if atoms.hydrogen == None or atoms.hydrogen.numAtoms() < 30: # if there is no hydrigens in PDB structure
                addHydrogens(pdb_name)
                pdb_name = pdb_name[:-4]+'_addH.pdb'
                atoms = parsePDB(pdb_name)
                LOGGER.info("Lack of hydrogen in the structure. Hydrogens will be added.")
        except: 
            LOGGER.info("Install Openbabel to add missing hydrigens or provide structure with hydrogens")
    
        Ligands = [] # Ligands can be more than one
        my_mol = PDBComplex()
        my_mol.load_pdb(pdb_name) # Load the PDB file into PLIP class
        select = select+' and not (resname '+' '.join(ignore_ligs)+')'
        ligand_select = atoms.select(select)
        analyzedLigand = []
        LOGGER.info("Detected ligands: ")
        for i in range(len(ligand_select.getResnums())): # It has to be done by each atom
            try:
                ResID = ligand_select.getResnames()[i]
                ChainID = ligand_select.getChids()[i]
                ResNames = ligand_select.getResnums()[i]
                my_bsid = str(ResID)+':'+str(ChainID)+':'+str(ResNames)
                if my_bsid not in analyzedLigand: 
                    LOGGER.info(my_bsid)
                    analyzedLigand.append(my_bsid)
                    my_mol.analyze()
                    my_interactions = my_mol.interaction_sets[my_bsid] # Contains all interaction data      
                    Ligands.append(my_interactions)
            except: 
                LOGGER.info(my_bsid+" not analyzed")

        return Ligands, analyzedLigand

    except:
        LOGGER.info("Install Openbabel and PLIP.")

def showLigandInteractions(PLIP_output):
    """Create a list of interactions from PLIP output created using calcLigandInteractions().
    Results can be displayed in VMD. 
    
    :arg PLIP_output: Results from PLIP program for protein-ligand interactions.
    :type PLIP_output: PLIP object obtained from calcLigandInteractions() 
    
    Note that 5 types of interactions are considered: hydrogen bonds, salt bridges, pi-stacking,
    cation-pi, hydrophobic and water bridges."""
    
    Inter_list_all = []
    for i in PLIP_output.all_itypes:
        param_inter = [method for method in dir(i) if method.startswith('_') is False]
        
        LOGGER.info(str(type(i)).split('.')[-1].strip("'>"))
        
        if str(type(i)).split('.')[-1].strip("'>") == 'hbond':
            Inter_list = ['hbond',i.restype+str(i.resnr), i[0].type+'_'+str(i.d_orig_idx), i.reschain,
                          i.restype+str(i.resnr_l), i[2].type+'_'+str(i.a_orig_idx), i.reschain_l, 
                          i.distance_ad, i.angle, i[0].coords, i[2].coords]
     
        if str(type(i)).split('.')[-1].strip("'>") == 'saltbridge':
            Inter_list = ['saltbridge',i.restype+str(i.resnr), '_'.join(map(str,i.negative.atoms_orig_idx)), i.reschain,
                          i.restype+str(i.resnr_l), '_'.join(map(str,i.positive.atoms_orig_idx)), i.reschain_l, 
                          i.distance, None, i.negative.center, i.positive.center]
                 
        if str(type(i)).split('.')[-1].strip("'>") == 'pistack':
             Inter_list = ['pistack',i.restype+str(i.resnr), '_'.join(map(str,i[0].atoms_orig_idx)), i.reschain,
                          i.restype+str(i.resnr_l), '_'.join(map(str,i[1].atoms_orig_idx)), i.reschain_l, 
                          i.distance, i.angle, i[0].center, i[1].center]           
        
        if str(type(i)).split('.')[-1].strip("'>") == 'pication':
             Inter_list = ['pication',i.restype+str(i.resnr), '_'.join(map(str,i[0].atoms_orig_idx)), i.reschain,
                          i.restype+str(i.resnr_l), '_'.join(map(str,i[1].atoms_orig_idx)), i.reschain_l, 
                          i.distance, None, i[0].center, i[1].center]                       
        
        if str(type(i)).split('.')[-1].strip("'>") == 'hydroph_interaction':
            Inter_list = ['hydroph_interaction',i.restype+str(i.resnr), i[0].type+'_'+str(i[0].idx), i.reschain,
                          i.restype+str(i.resnr_l), i[2].type+'_'+str(i[2].idx), i.reschain_l, 
                          i.distance, None, i[0].coords, i[2].coords]           
             
        if str(type(i)).split('.')[-1].strip("'>") == 'waterbridge':
            water = i.water
            Inter_list = ['waterbridge',i.restype+str(i.resnr), i[0].type+'_'+str(i[0].idx), i.reschain,
                          i.restype+str(i.resnr_l), i[3].type+'_'+str(i[3].idx), i.reschain_l, 
                          [i.distance_aw, i.distance_dw], [i.d_angle, i.w_angle], i[0].coords, i[3].coords, 
                          i.water.coords, i[7].residue.name+'_'+str(i[7].residue.idx)]    
        else: pass
                      
        Inter_list_all.append(Inter_list)               
        
    return Inter_list_all


def showProteinInteractions_VMD(atoms, interactions, color='red',**kwargs):
    """Save information about protein interactions to a TCL file (output)
    which can be further use in VMD program to display all intercations in a graphical interface
    (in TKConsole: play script_name.tcl).
    Different types of interactions can be saved separately (color can be selected) 
    or all at once for all types of interactions (hydrogen bonds - blue, salt bridges - yellow,
    pi stacking - green, cation-pi - orange and hydrophobic - silver).
    
    :arg atoms: an Atomic object from which residues are selected
    :type atoms: :class:`.Atomic`
    
    :arg interactions: List of interactions for protein interactions.
    :type interactions: List of lists
    
    :arg color: color to draw interactions in VMD program,
                not used only for single interaction type.
    :type color: str or **None**, by default `red`.
    
    :arg output: name of TCL file where interactions will be saved.
    :type output: str
        
    Example (hydrogen bonds for protein only): 
    >> interactions = calcHydrogenBonds(atoms.protein, distA=3.2, angle=30)
    or all interactions at once:
    >> interactions = calcProteinInteractions(atoms.protein) """

    try:
        coords = (atoms._getCoords() if hasattr(atoms, '_getCoords') else
                    atoms.getCoords())
    except AttributeError:
        try:
            checkCoords(coords)
        except TypeError:
            raise TypeError('coords must be an object '
                            'with `getCoords` method')

    if not isinstance(interactions, list):
        raise TypeError('interactions must be a list of interactions.')
    
    try:
        output = kwargs['output']
    except:
        output = atoms.getTitle()+'_interaction.tcl'
    
    tcl_file = open(output, 'w') 
    
    
    def TCLforSingleInteaction(interaction, color='blue', tcl_file=tcl_file):
        """Protein interactions from ProDy for a single Interaction"""
        
        tcl_file.write('draw color '+color+'\n')
        for nr_i,i in enumerate(interaction):
            at1 = atoms.select('index '+' '.join([k for k in i[1].split('_') if k.isdigit() ] ))
            at1_atoms = ' '.join(map(str,list(calcCenter(at1.getCoords()))))
            at2 = atoms.select('index '+' '.join([kk for kk in i[4].split('_') if kk.isdigit() ] ))
            at2_atoms = ' '.join(map(str,list(calcCenter(at2.getCoords()))))
                        
            tcl_file.write('draw line {'+at1_atoms+'} {'+at2_atoms+'} style dashed width 4\n')
            
            tcl_file.write('mol color Name\n')
            tcl_file.write('mol representation Licorice 0.100000 12.000000 12.000000\n')
            tcl_file.write('mol selection (resname '+at1.getResnames()[0]+' and resid '+str(at1.getResnums()[0])
                           +' and chain '+at1.getChids()[0]+' and noh) or (resname '+at2.getResnames()[0]+' and resid '
                           +str(at2.getResnums()[0])+' and chain '+at2.getChids()[0]+' and noh)\n')
            tcl_file.write('mol material Opaque\n')
            tcl_file.write('mol addrep 0 \n')
     
    if len(interactions) == 6:   
        # For all six types of interactions at once
        # HBs_calculations, SBs_calculations, SameChargeResidues, Pi_stacking, Pi_cation, Hydroph_calculations
        colors = ['blue', 'yellow', 'red', 'green', 'orange', 'silver']
        
        for nr_inter,inter in enumerate(interactions):
            TCLforSingleInteaction(inter, color=colors[nr_inter], tcl_file=tcl_file)

    elif len(interactions[0]) == 0 or interactions == []:
        LOGGER.info("Lack of results")
        
    else:
        TCLforSingleInteaction(interactions,color)

    tcl_file.write('draw materials off')
    tcl_file.close()   
    LOGGER.info("TCL file saved")


def showLigandInteraction_VMD(atoms, interactions, **kwargs):
    """Save information from PLIP program for ligand-protein interactions in a TCL file
    which can be further used in VMD program to display all intercations in a graphical 
    interface (hydrogen bonds - `blue`, salt bridges - `yellow`,
    pi stacking - `green`, cation-pi - `orange`, hydrophobic - `silver` and water bridges - `cyan`).
    
    :arg atoms: an Atomic object from which residues are selected
    :type atoms: :class:`.Atomic`
    
    :arg interactions: List of interactions for protein-ligand interactions.
    :type interactions: List of lists
    
    :arg output: name of TCL file where interactions will be saved.
    :type output: str

    To obtain protein-ligand interactions:
    >> calculations = calcLigandInteractions(atoms)
    >> interactions = showLigandInteractions(calculations) """

    try:
        coords = (atoms._getCoords() if hasattr(atoms, '_getCoords') else
                    atoms.getCoords())
    except AttributeError:
        try:
            checkCoords(coords)
        except TypeError:
            raise TypeError('coords must be an object '
                            'with `getCoords` method')

    if not isinstance(interactions, list):
        raise TypeError('interactions must be a list of interactions.')
    
    try:
        output = kwargs['output']
    except:
        output = atoms.getTitle()+'_interaction.tcl'
    
    tcl_file = open(output, 'w') 
    
    if len(interactions[0]) >= 10: 
        dic_color = {'hbond':'blue','pistack':'green','saltbridge':'yellow','pication':'orange',
                     'hydroph_interaction':'silver','waterbridge':'cyan'}
        
        for i in interactions:
            tcl_file.write('draw color '+dic_color[i[0]]+'\n')
            
            if i[0] == 'waterbridge':
                hoh_id = atoms.select('x `'+str(i[11][0])+'` and y `'+str(i[11][1])+'` and z `'+str(i[11][2])+'`').getResnums()[0]
                tcl_file.write('draw line {'+str(' '.join(map(str,i[9])))+'} {'+str(' '.join(map(str,i[11])))+'} style dashed width 4\n')
                tcl_file.write('draw line {'+str(' '.join(map(str,i[10])))+'} {'+str(' '.join(map(str,i[11])))+'} style dashed width 4\n')
                tcl_file.write('mol color Name\n')
                tcl_file.write('mol representation Licorice 0.100000 12.000000 12.000000\n')
                tcl_file.write('mol selection (resname '+i[1][:3]+' and resid '+str(i[1][3:])
                               +' and chain '+i[3]+' and noh) or (resname '+i[4][:3]+' and resid '
                               +str(i[4][3:])+' and chain '+i[6]+' and noh) or (water and resid '+str(hoh_id)+')\n')
                
            else:
                tcl_file.write('draw line {'+str(' '.join(map(str,i[9])))+'} {'+str(' '.join(map(str,i[10])))+'} style dashed width 4\n')
                tcl_file.write('mol color Name\n')
                tcl_file.write('mol representation Licorice 0.100000 12.000000 12.000000\n')
                tcl_file.write('mol selection (resname '+i[1][:3]+' and resid '+str(i[1][3:])
                               +' and chain '+i[3]+' and noh) or (resname '+i[4][:3]+' and resid '
                               +str(i[4][3:])+' and chain '+i[6]+' and noh)\n')
            tcl_file.write('mol material Opaque\n')
            tcl_file.write('mol addrep 0 \n')            


    tcl_file.write('draw materials off')
    tcl_file.close()   
    LOGGER.info("TCL file saved")


def showHydrogenBondsMap(atoms, distA=3.0, angle=40, cutoff_dist=20, **kwargs): 
    """Show distribution of hydrogen bonds in protein structure.
    using :func:`~matplotlib.pyplot.plot`. 
    
    :arg atoms: an Atomic object from which residues are selected
    :type atoms: :class:`.Atomic`
    
    :arg distA: non-zero value, maximal distance between donor and acceptor.
    :type distA: int or float, default is 3.0.
    
    :arg angle: non-zero value, maximal (180 - D-H-A angle) (donor, hydrogen, acceptor).
    :type distA: int or float, default is 40.
    
    :arg cutoff_dist: non-zero value, interactions will be found between residues that are higher than cutoff_dist.
    :type cutoff_dist: int, default is 20 atoms.

    Structre should contain hydrogens.
    If not they can be added using addHydrogens(pdb_name) function availabe in ProDy after Openbabel instalation.
    `conda install -c conda-forge openbabel`
    
    Note that the angle which is considering is 180-defined angle D-H-A (in a good agreement with VMD)
    Results can be displayed in VMD by using showVMDinteraction() """
    
    import matplotlib
    import matplotlib.pyplot as plt

    try:
        coords = (atoms._getCoords() if hasattr(atoms, '_getCoords') else
                    atoms.getCoords())
    except AttributeError:
        try:
            checkCoords(coords)
        except TypeError:
            raise TypeError('coords must be an object '
                            'with `getCoords` method')

    HBS_calculations = calcHydrogenBonds(atoms.protein, **kwargs)
    HBsMap = np.zeros([atoms.select('name CA').numAtoms(),atoms.select('name CA').numAtoms()]) 
    resIDs = list(atoms.select('name CA').getResnums())
    resChIDs = list(atoms.select('name CA').getChids())
    resIDs_with_resChIDs = list(zip(resIDs, resChIDs))

    for i in HBS_calculations:
        m1 = resIDs_with_resChIDs.index((int(i[0][3:]),i[2]))
        m2 = resIDs_with_resChIDs.index((int(i[3][3:]),i[5]))
        HBsMap[m1][m2] = HBsMap[m1][m2] + 1.0

    showAtomicMatrix(HBsMap, atoms=atoms.select('name CA'), cmap=plt.cm.Greys, interpolation='none', vmin=0, vmax=1.0)
    plt.xlabel('Residue')
    plt.ylabel('Residue')
    #plt.tight_layout()
    if SETTINGS['auto_show']:
        showFigure()
    return plt.show


def showInteractionMap(atoms):
    """Show distribution of protein interactions with default paramaters and score as follows:
        (1) Hydrogen bonds +2
        (2) Salt Bridges +3
        (3) Repulsive Ionic Bonding -1
        (4) Pi stacking interactions +3
        (5) Pi-cation interactions +3
        (6) Hydrophobic interactions +1
        and display sum of interation per residue. 
        
    :arg atoms: an Atomic object from which residues are selected
    :type atoms: :class:`.Atomic`"""

    import matplotlib
    import matplotlib.pyplot as plt

    try:
        coords = (atoms._getCoords() if hasattr(atoms, '_getCoords') else
                    atoms.getCoords())
    except AttributeError:
        try:
            checkCoords(coords)
        except TypeError:
            raise TypeError('coords must be an object '
                            'with `getCoords` method')

    InteractionsMap = calcInteractionMatrix(atoms)
    showAtomicMatrix(InteractionsMap, atoms=atoms.select('name CA'), interpolation='none', cmap='seismic'); plt.clim([-3,3])
    plt.xlabel('Residue')
    plt.ylabel('Residue')
    #plt.tight_layout()
    if SETTINGS['auto_show']:
        showFigure()
    return plt.show

