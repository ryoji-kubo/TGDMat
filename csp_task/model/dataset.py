import torch
import os
import bz2
import pickle
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.data import Data
from model.data_utils import (preprocess, add_scaled_lattice_prop, get_short_prompt_variant_key)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





class MaterialDataset(Dataset):
    def __init__(
        self,
        data_dir,
        dataset,
        prompt_type,
        file,
        short_prompt_training_scheme='full',
        short_prompt_eval_omit=None,
    ):
        super().__init__()
        self.dataset = dataset
        self.path = os.path.join(data_dir, dataset, file + ".csv")
        self.df = pd.read_csv(self.path)
        self.prompt_type = prompt_type
        self.short_prompt_training_scheme = short_prompt_training_scheme
        self.short_prompt_eval_omit = short_prompt_eval_omit
        if self.dataset == 'perov_5':
            self.prop = ['heat_ref']
            all_attributes = ["pretty_formula", "elements", "heat_ref", "spacegroup", "system"]
        elif self.dataset == 'carbon_24':
            self.prop = ["energy_per_atom"]
            all_attributes = ["pretty_formula", "elements", "energy_per_atom", "spacegroup", "system"]
        elif self.dataset == 'mp_20':
            self.prop = ["formation_energy_per_atom", "band_gap", "e_above_hull"]
            all_attributes = ["pretty_formula", "elements", "formation_energy_per_atom", "band_gap", "e_above_hull",
                              "spacegroup", "system"]

        self.niggli = True
        self.primitive = False
        self.graph_method = 'crystalnn'
        self.lattice_scale_method = 'scale_length'
        self.preprocess_workers = 30

        # print(self.path)
        # print(dataset)
        # print(file)
        cache_path = os.path.join(data_dir, dataset, file + "_short_prompt_variants_v1.pbz2")
        # print(cache_path)

        if os.path.exists(cache_path):
            print("Loading data...", end=" ", flush=True)
            with bz2.BZ2File(cache_path, "rb") as f:
                self.cached_data = pickle.load(f)
            print("done.")
        else:
            self.cached_data = preprocess(self.path,
                                      niggli=self.niggli,
                                      primitive=self.primitive,
                                      graph_method=self.graph_method,
                                      prop_list=self.prop,
                                      all_attributes = all_attributes,
                                      dataset = self.dataset)
            with bz2.BZ2File(cache_path, "wb") as f:
                pickle.dump(self.cached_data, f)



        add_scaled_lattice_prop(self.cached_data, self.lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]
        (frac_coords, atom_types, lengths, angles, edge_indices,to_jimages, num_atoms) = data_dict['graph_arrays']

        one_hot = np.zeros((len(atom_types), 100))
        for i in range(len(atom_types)):
            one_hot[i][atom_types[i]-1]=1

        if self.prompt_type=='long':
            text = data_dict['text_long']
        else:
            text = self._get_short_prompt_embedding(data_dict)

        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            atom_types_one_hot=torch.Tensor(one_hot),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,
            text=text
        )
        return data

    def _get_short_prompt_embedding(self, data_dict):
        if 'text_short_variants' not in data_dict:
            return data_dict['text_short']

        if self.short_prompt_training_scheme == 'leave_one_out' and self.short_prompt_eval_omit is None:
            variant_keys = [get_short_prompt_variant_key()]
            variant_keys.extend(
                get_short_prompt_variant_key(attr)
                for attr in data_dict.get('short_prompt_attributes', [])
            )
            return data_dict['text_short_variants'][random.choice(variant_keys)]

        variant_key = get_short_prompt_variant_key(self.short_prompt_eval_omit)
        return data_dict['text_short_variants'].get(variant_key, data_dict['text_short'])


def main():
    return None


if __name__ == "__main__":
    main()
