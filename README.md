# DALAUP
If you find this method helpful for your research, please cite this paper:

```latex
@inproceedings{cheng2019deep,
  author    = {Anfeng Cheng and
               Chuan Zhou and
               Hong Yang and
               Jia Wu and
               Lei Li and
               Jianlong Tan and
               Li Guo},
  title     = {Deep Active Learning for Anchor User Prediction},
  booktitle = {The 28th International Joint Conference on Artificial Intelligence(IJCAI-19)},
  year      = {2019},
  url       = {https://www.ijcai.org/proceedings/2019/0298.pdf}
}
```

------

### Requirement

- python >= 3.6
- pytorch >= 0.4
- numpy
- tqdm
- networkx

------

### Dataset

The dataset used in this paper can be obtained from the original papers.

| Dataset                | Paper                                                        |
| ---------------------- | ------------------------------------------------------------ |
| Foursquare and Twitter | Xiangnan Kong, Jiawei Zhang, and Philip S Yu. Inferring anchor links across multiple heterogeneous social networks. In Proceedings of the 22nd ACM international conference on Information & Knowledge Management, pages 179–188. ACM, 2013.<br>Zhang J, Kong X, Philip S Y. Predicting social links for new users across aligned heterogeneous social networks[C]//2013 IEEE 13th International Conference on Data Mining. IEEE, 2013: 1289-1294. |

Dataset: https://github.com/ColaLL/IONE

---

### How to use

#### Step 1. Structural Context Extract

Extract structural context for each users.

```bash
python feature_extract.py --data_path 'XXX' --save_file_path 'XXX' --restart_probability 0.6
```

* The _data_path_ is the network structural data formulated with adj list (e.g. line1:user1_id, user2_id  line2:user1_id user3_id). 

* The _save_file_path_ is the extracted context. Each line donates the feature for one user ( index is user ID). (\mathbf{m}_{A}^{i}​ or \mathbf{m}_{B}^{i}​ in paper) 

* The _restart_probability_ is the restart probability.

  **Note** : The user id must be int, continuous and must start from 0. The anchor user pair must have the same id in two network and they nust start with 0 (e.g. If the total number of anchor user is 100, the anchor ids are from 0 to 99).

#### Step 2. Representation of Pairs of Users Across Networks and Anchor User Pair Classiﬁcation

Two **separate** parts, both can predict anchor users.

**AUP:** Evaluate by cosine similarity.

```bash
python model.py --feature_A 'XXX' --feature_B 'XXX' --total_anchor NUM --train_ratio 0.5 --gpu_id 4
```

- The *feature_A/feature_B* is the file of users' structural context.
-  The *total_anchor* is total number of anchor users.

**AUP:** Evaluate by classiﬁcation.

​	Set the optional parameters *is_classification=True(default=False)* .

**Note:**  Anchor users can be predicted with above steps(Recommendatory: Evaluate by cosine similarity).

---

### Disclaimer

If you have any problems about the paper or the code,  please report them to me. Feel free to contact chenganfeng AT iie.ac.cn.

We have updated the Fig.2 in our paper, we made a new typesetting on the figures in camera-ready and confused two evaluation metrics in Fig.2. We correct this in the revision which can be found in [arxiv](https://arxiv.org/abs/1906.07318). Thanks the PHD student Rui Tang from sichuan university for pointing this. (2019-11-07 Anfeng Cheng)

