# StructGNN: An Efficient Graph Neural Network Framework for Static Structural Analysis

_Authors: Yuan-Tung Chou, Wei-Tze Chang, Jimmy G. Jean, Kai-Hung Chang, Yin-Nan Huang, Chuin-Shan Chen_

In the field of structural analysis prediction via supervised learning, neural networks are widely employed. Recent advances in graph neural networks (GNNs) have expanded their capabilities, enabling the prediction of structures with diverse geometries by utilizing graph representations and GNNs’ message-passing mechanism. However, conventional message-passing in GNNs doesn’t align with structural properties, resulting in inefficient computation and limited generalization to extrapolated datasets. To address this, a novel structural graph representation, incorporating pseudo nodes as rigid diaphragms in each story, alongside an efficient GNN framework called StructGNN is proposed. StructGNN employs an adaptive message-passing mechanism tailored to the structure’s story count, enabling seamless transmission of input loading features across the structural graph. Extensive experiments validate the effectiveness of this approach, achieving over 99% accuracy in predicting displacements, bending moments, and shear forces. StructGNN also exhibits strong generalization over non-GNN models, with an average accuracy of 96% on taller, unseen structures. These results highlight StructGNN’s potential as a reliable, computationally efficient tool for static structural response prediction, offering promise for addressing challenges associated with dynamic seismic loads in structural analysis.

Keywords: deep learning, graph neural network, graph representation, structural analysis


### Data
The data includes 2000 linear static analysis responses for 2-6 story structures and 500 linear static analysis responses for 7-9 story structures. Each analysis data is packaged as a `graph.pt` where the structure geometries and responses for each node are saved in this PyTorch object.
![linear static analysis data](figures/generated_structures.png)


### StructGNN model
StructGNN model is composed of an encoder, message-passing layers, and a decoder. The number of the message-passing layer varies with the story number of the structure. Structures with `N` story will go through `N+1` message-passing layers to ensure the external force information applied on the roof is passed through the entire graph.
![model architecture](figures/model_architecture.png)
![deformable layer](figures/deformable_layer.png)

### Results
After training, StructGNN can predict responses in 2-6 story structures with over 99% in displacement, shear force, and bending moment. In addition, when testing on 7-9 story structures, which are taller and the model has never seen before, StrcutGNN can still have 95% of accuracy.

We further visualized the node embedding propagation in each stage of the message-passing layers. If we don't use the mechanism that the layer number will vary with the structure's story number, the performance is not as good, since using a fixed number of message-passing layers can either be too much for a relatively short structure, or too few for a tall structure. The embeddings of using a fixed number message-passing layer are visualized below.
![fixed number propagation](figures/fixed_layer_number_propagation.png)

By using the dynamic number of message-passing layers, the features of the external forces can be propagated to the entire structure properly, making it even generalizable on taller, unseen structures.
![customized number propagation](figures/customized_propagation.png)

### Citation
Y-T Chou, W-T Chang, JG Jean, K-H Chang, Y-N Huang, C-S Chen (2024), "StructGNN: an efficient graph neural network framework for static structural analysis," Computers and Structures, accepted. (IF 4.7 (2022), ENGINEERING, CIVIL 26/139, COMPUTER SCIENCE, INTERDISCIPLINARY APPLICATIONS 35/110).

