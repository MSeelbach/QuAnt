# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 09:47:43 2022

@author: Marcel
"""


#Code is based on https://docs.ocean.dwavesys.com/projects/system/en/latest/_modules/dwave/system/composites/tiling.html


import networkx as nx
import dwave_networkx as dnx
import matplotlib.pyplot as plt   # doctest: +SKIP


import neal

import dimod

import dwave_networkx as dnx
from dwave.system import FixedEmbeddingComposite
from dwave.system import DWaveSampler

import pickle


class TilingCom:

    def __init__(self):
    
        self.qpu_sampler=DWaveSampler( solver='Advantage_system4.1')# solver={'qpu': True,'topology__type': 'pegasus'})#,  solver='Advantage_system5.1' if currently accessible
   
#        pickle.dump(self.qpu_sampler.properties(), open("SolverProp.p", "wb" ) ) #Save solver information
    
    def GetTilingEmbedding(self): 
    
            G = dnx.pegasus_graph(3)
            
            sub_m= 2
            sub_n=2
            
            t= 4
            
            tile = dnx.chimera_graph(sub_m, sub_n, t)
            
            
            qpu_sampler= self.qpu_sampler
            
            
            n=m=qpu_sampler.properties['topology']['shape'][0] - 1
            num_sublattices=3
            edges_per_cell=t**2 +t 
            
            nodes_per_cell = t * 2
            
            
            
            system =  dnx.pegasus_graph(m,
                                           node_list=qpu_sampler.structure.nodelist,
                                           edge_list=qpu_sampler.structure.edgelist)
                        #Vector specification in terms of nice coordinates:
            c2i = {dnx.pegasus_coordinates(m+1).linear_to_nice(linear_index):
                   linear_index for linear_index in system.nodes()}
            
            sub_c2i = {chimera_index: linear_index for (linear_index, chimera_index)
                   in tile.nodes(data='chimera_index')}
            
            
            def _between(qubits1, qubits2):
                edges = [edge for edge in system.edges if edge[0] in qubits1
                         and edge[1] in qubits2]
                return len(edges)
            
            # Get the list of qubits in a cell
            def _cell_qubits(s, i, j):
                return [c2i[(s, i, j, u, k)] for u in range(2) for k in range(t)
                        if (s, i, j, u, k) in c2i]
            
            
            cells = [[[False for _ in range(n)] for _ in range(m)]
                             for _ in range(num_sublattices)]
                    
            qubitsList=[]
            for s in range(num_sublattices):
                for i in range(m):
                    for j in range(n):
                        qubits = _cell_qubits(s, i, j)
                        cells[s][i][j] = (
                            len(qubits) == nodes_per_cell
                            and _between(qubits, qubits) == edges_per_cell)
                        
                        if cells[s][i][j]==True:
                            qubitsList.extend( _cell_qubits(s,i,j))
            
            
            
            embeddings=[]
            
            for s in range(num_sublattices):
                        for i in range(m + 1 - sub_m):
                            for j in range(n + 1 - sub_n):
                                
                                # Check if the sub cells are matched
                                match = all(cells[s][i + sub_i][j + sub_j]
                                            for sub_i in range(sub_m)
                                            for sub_j in range(sub_n))
                                
                                # Check if there are connections between the cells.
                                # Both Pegasus and Chimera have t vertical and t horizontal between cells:
                                for sub_i in range(sub_m):
                                    for sub_j in range(sub_n):
                                        if sub_m > 1 and sub_i < sub_m - 1:
                                            match &= _between(
                                                _cell_qubits(s, i + sub_i, j + sub_j),
                                                _cell_qubits(s, i + sub_i + 1, j + sub_j)) == t
                                        if sub_n > 1 and sub_j < sub_n - 1:
                                            match &= _between(
                                                _cell_qubits(s, i + sub_i, j + sub_j),
                                                _cell_qubits(s, i + sub_i, j + sub_j + 1)) == t
                                
                                if match:
                                    # Pull those cells out into an embedding.
                                    embedding = {}
                                    for sub_i in range(sub_m):
                                        for sub_j in range(sub_n):
                                            cells[s][i + sub_i][j + sub_j] = False  # Mark cell as matched
                                            for u in range(2):
                                                for k in range(t):
                                                    embedding[sub_c2i[sub_i, sub_j, u, k]] = [
                                                        c2i[(s,i + sub_i, j + sub_j, u, k)]]
            
                                    embeddings.append(embedding)
            
            self.embeddings= embeddings
            self.tile=tile
            
            
            return embeddings, tile 
                

    def runSimulated(self,W ):
                
                
                    

                    
            
            
                structured_sampler = dimod.StructureComposite(neal.SimulatedAnnealingSampler(),
                                                               self.qpu_sampler.nodelist, self.qpu_sampler.edgelist)
            
                embedding = {}
            
            
                for g in list( self.qpu_sampler.nodelist):
                     embedding[g]=[g]    
            
            
                
                number = W.shape[0]
                dic={}
            
                for s in range(number):
                    
                    mapping=self.embeddings[s]
                    
                    for t in list(self.tile.edges):
                            
                            
                        if W[s,  t[0]  ,  t[1]  ]!=0:
                            dic[( mapping[ t[0]][0]  , mapping[ t[1]][0] )]= 2*W[s,  t[0]  ,  t[1]  ]       
                   
                    
                    for n in list(self.tile.nodes):
                        if W[s , n ,  n  ]!=0:

                            dic[(mapping[n][0] ,mapping[n][0])]= W[s , n ,  n  ] 
        
                sampler = FixedEmbeddingComposite(structured_sampler, embedding)
            
                sampleset = sampler.sample_qubo(dic ,num_reads=100)
                return sampleset
            
            
    def runrealDeal(self,W,descr ):
                
                
                    
                structured_sampler = dimod.StructureComposite( self.qpu_sampler,
                                                               self.qpu_sampler.nodelist, self.qpu_sampler.edgelist)
            
                embedding = {}
            
            
                for g in list( self.qpu_sampler.nodelist):
                     embedding[g]=[g]    
            
            
                
                number = W.shape[0]
                dic={}
            
                for s in range(number):
                    
                    mapping=self.embeddings[s]
                    
                    for t in list(self.tile.edges):
                            
                            
                        if W[s,  t[0]  ,  t[1]  ]!=0:
                            dic[( mapping[ t[0]][0]  , mapping[ t[1]][0] )]= 2*W[s,  t[0]  ,  t[1]  ]       
                   
                    
                    for n in list(self.tile.nodes):
                        if W[s , n ,  n  ]!=0:

                            dic[(mapping[n][0] ,mapping[n][0])]= W[s , n ,  n  ] 
        
                sampler = FixedEmbeddingComposite(structured_sampler, embedding)
            
                sampleset = sampler.sample_qubo(dic ,num_reads=100)
                
                #Result=[]
                #for datum in sampleset.data(['sample', 'energy', 'num_occurrences','chain_break_fraction']):   
                               #print(datum.sample, "Energy: ", datum.energy, "Occurrences: ", datum.num_occurrences)
                               #Result.append([datum.sample,  datum.energy,  datum.num_occurrences, datum.chain_break_fraction])
   
                #pickle.dump([Result[0:2],W], open( "saveExecutions"+descr+".p", "wb" ) )
                return sampleset

            
    