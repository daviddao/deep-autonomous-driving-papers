<div align="center">
  <img src="http://i.imgur.com/pEeCAS4.jpg"><br><br>
</div>
-----------------

# Roadmap to Deep Autonomous Driving

> This is a curated list of Deep Learning for Autonomous Driving

> It serves as: 

> 1. *reading roadmap* from outline to detail and focuses on state-of-the-art research with applications in the wild
> 2. *toolkit collection* and provides links to important datasets, software and researchers in this field

---------------------------------------

## Why **Deep** Autonomous Driving?

Several reasons for:

- Robustness: Classical AD techniques require intensive hand engineering, road modeling and special case handeling.
- Deep Learning, on the other side, is data centric and require minimal hand-engineering. 
- Salient sensory features determined for specific driving conditions
- It is easy to fuse different sensory inputs as it is learned
- End-to-end removes the need for hand-crafted heuristics and ill-posed problems (i.e like computing depth from stereo images) 
- Might learn new and more useful features

A future system should include map building, visual odometry, spatial reasoning, path finding and other strategies for the identification of traversable area. 

## Major challenges in Deep AD

> Unavailabiliy of large diverse datasets with consistent driver behaviour
> How to measure performance robustness?

## 1. Planning for Deep Autonomous Driving

> There are three paradigms for vision-based autonomous driving systems: 

> 1. *Mediated Perception* parses every scene into structured data and derives decision from that
> 2. *Behaviour Reflex* maps directly from input image to driving control using deep learning 
> 3. *Direct Perception* maps input to a small number of perception indicators and make a driving decision

### 1.1 Mediated Perception 

#### An Empirical Evaluation of Deep Learning on Highway Driving (B Huval, T Wang, S Tandon, J Kiske, W Song, J Pazhayampallil, M Andriluka, P Rajpurkar, T Migimatsu, R Cheng-Yue, F Mujica, A Coates, AY Ng 2015)

> Uses Overfeat for real-time vehicle and lane detection. Due to ambiguity, the authors modified overfeat to first predict a object mask and then used a bounding box regression for the final object detection.

> - Architecture: Overfeat
> - Input: 640x480 input 
> - Output: Object mask

#### Fast Incremental Learning for Off-Road Robot Navigation (A Provodin, L Torabi, B Flepp, Y LeCun, M Sergio, LD Jackel, U Muller, J Zbontar, 2016)

> Uses transfer learning to cope with the problem of missing data in AD. Shows that a pretrained CNN on ImageNet can extract good features for AD.

> - Architecture: 3-layer CNN
> - Input: 59x59 image patches around pixels
> - Output: Each pixel as "drivable" or "obstacle" 

#### Instance-Level Segmentation for Autonomous Driving with Deep Densely Connected MRFs (Z Zhang, S Fidler, R Urtasun, 2016)

#### Deep Tracking: Seeing Beyond Seeing Using Recurrent Neural Networks (P Ondruska, I Posner, AAAI2016) :star:

[[code](https://github.com/pondruska/DeepTracking)]

> Introduces the DeepTracking framework. A RNN can be trained to track objects (even when occluded) only using the raw sensory input. 

> - Architecture: 3-layer CNN-GRU RNN
> - Input: Raw occupancy map
> - Output: Object map

> The authors use two tricks: Raw sensor input contains weak supervision and they predict future data points (instead present) to model movement.

#### Deep Tracking on the Move: Learning to Track the World from a Moving Vehicle using Recurrent Neural Networks (J Dequaire, D Rao, P Ondruska, DZ Wang, I Posner, 2016) :star:

> End-to-end approach for tracking objects in a moving car using the DeepTracking framework. It can track occluded objects using an RNN and account for the movement of the vehicle by adding spatial invariance.

> - Architecture: 3-layer CNN-GRU RNN with STN 
> - Input: Raw occupancy map
> - Output: Object map 

#### End-to-End Tracking and Semantic Segmentation Using Recurrent Neural Networks (P Ondruska, J Dequaire, DZ Wang, I Posner, 2016)

> DeepTracking trained with unsupervised and (weak) supervised data to perform tracking and semantic segmentation. Network learns implicit tracking and world states which can be used for segmentation using transfer learning.

> - Architecture: 3-layer CNN-GRU RNN with bias and dilated convolutions
> - Input: Raw occupancy map, semantic segmentation map
> - Output: Object map with semantic segmentation

> The author extends the DeepTracking architecture using dilated convolutions to track objects of different scales, dynamic memory (LSTM or GRU) for information caching, static memory which helps to store place-specific information

#### Find Your Own Way: Weakly-Supervised Segmentation of Path Proposals for Urban Autonomy (D Barnes, W Maddern, I Posner, 2016)

> Creates more path segmentation training data by combining future frames and sensor information. Uses a SegNet to train and predict paths using this additional data. Evaluates on KITTI and Oxford Car dataset.

> - Architecture: SegNet
> - Input: 640x256 image
> - Output: Segmented image with path, unknown, obstacle

### 1.2 Behaviour Reflex (Supervised End-to-End Learning)

> General Design: Input -> NN -> Control

#### ALVINN, An Autonomous Land Vehicle in a Neural Network (DA Pomerleau 1988) :star:

> First seminal work of a neural network for autonomous driving, trained on simulated data and achieving 90% direction prediction accuracy on road following.

> - Architecture: 2-layer FCN 
> - Input: 30x32 image, 8x32 laser range (Simulated)
> - Output: Direction 

> During testing, intensity output is recirculated to the input unit, which improves accuracy.

#### Off-Road Obstacle Avoidance through End-to-End Learning (Y LeCun, U Mueller, J Ben, E, Cosatto, B Flepp 2005) :star:

> First use of a CNN for autonomous driving, introduces DAVE an off-road autonomous vehicle trained end-to-end to avoid obstacles. 

> - Architecture: 6-layer CNN 
> - Input: left/right pair of 149x58 image 
> - Output: Steering angle 

#### Learning Long-Range Vision for Autonomous Off-Road Driving (R Hadsell, P Sermanet, J Ben, A Erkan, M Scoffier, K Kavukcuoglu, U Mueller, Y LeCun 2009)

> Building upon DAVE, it describes DARPA's LAGR.


#### Evolving large-scale neural networks for vision-based torcs (J Koutnik, G Cuccu, J Schmidhuber, FJ Gomez 2013)

> Using evolutionary networks to train a car in TORCS.

 
#### End to End Learning for Self-Driving Cars (M Bojarski, DD Testa, D Dworakowski, B Firner, B Flepp, P Goyal, LD Jackel, M Monfort, U Muller, J Zhang, X Zhang, J Zhao, K Zieba 2016) :star:

> DAVE-2 trains a CNN end-to-end and has a 98% autonomy rate. The authors show that just by using steering as sparse signal and 100h driving, the CNN is able to learn useful features (ie outline of the road).
> The authors evaluated driving on highway and roads under different weather circumstances, without lane switches and road changes.

> - Architecture: 9-layer CNN (chosen empirically)
> - Input: 200x66 images sampled from Video (10FPS)
> - Output: Inverse turning radius

> Represents steering command as *1/r*, where *r* is the turning radius in meters. 
> Train the car with negative examples using shifted cameras.
> Ranking after autonomous score: Each intervention is a penalty of 6 seconds.

### 1.3 Direct Perception

> General Design: Input -> NN -> Indicators -> Controller -> Control

#### DeepDriving: Learning Affordance for Direct Perception in Autonomous Driving (C Chen, A Seff, A Kornhauser, J Xiao in ICCV2015) :star:

> Describes well pro and cons of mediated and end-to-end learning and introduces another approach: direct perception. Uses a CNN to learn affordance values for a controller. Focuses on Highway Driving with multiple lanes.

> - Architecture: AlexNet  
> - Input: 280x210 image (simulated)
> - Output: 13 affordance indicators 


### 1.4 Reinforcement Learning

> Uses policy to estimate the best action in a given state

#### Query-Efficient Imitation Learning for End-to-End Autonomous Driving (J Zhang, K Cho, 2016)

> A human driver (reference policy) cannot cover all situations in data. This paper introduces imitation learning for AD, where a CNN learns a primary policy and together with the reference policy iterate to generate more data.
> A safety policy, estimated by an additional FCN, predicts, if it is safe for a NN to drive.

> - Architecture: 6-layer CNN, 2 layer FCN 
> - Input: 160x72 image (simulated in TORCS), Conv5
> - Output: Steering angle, safe/unsafe

#### Watch This: Scalable Cost-Function Learning for Path Planning in Urban Environments (M Wulfmeier, DZ Wang, I Posner, IROS16 Best Student Paper)

> Extends Maximum Entropy Inverse RL by using a multi-scale F-CNN architecture trained on 25000 trajectories. The trained network is robust towards sensor miscalibration and outperforms hand-designed features

> - Architecture: 9-layer Fully-CNN with Max Pool and Multi-Scale Architecture 
> - Input: 15m long trajectories on a 2D based 50x50m static map with 0.25m resolution (LIDAR)
> - Output: Costmap


---------------------------------------

## 2 Theory


#### Maximum Entropy Deep Inverse Reinforcement Learning (M Wulfmeier, P Ondruska, I Posner, NIPS RL Workshop 2015) 

> Maximum Entropy IRL models suboptimal samples (i.e driver trajectories) by assign each sample a probability proportional to its expected reward. The authors then use a CNN to predict costs for large state spaces and complex reward structures, achieving state-of-the-art.


### Traditional Trajectory Planning

- Trajectory Planning for Bertha - A Local, Continous Method (Ziegler et al. 2014)
- Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame (Werling et al. 2010)


### Curriculum Learning 



## Appendix A: Deep Learning Architectures

- FCN
- CNN (Convolutional Neural Network)
- Fully-CNN
- AlexNet
- VGG
- ResNet
- SegNet
- OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks (Sermanet et al. 2014)
- Faster R-CNN 


### 3 Important Researchers

- Sergey Levine
- Peter Abbeel
- Raquel Urtasun 
- Yann LeCun
- Ingmar Posner
- Urs Muller


### Conferences & Workshops and Groups related to Deep Learning in AD

- Deep Learning for Action and Interaction NIPS 2016
- Reliable Machine Learning in the Wild [[Link](https://sites.google.com/site/wildml2016nips/)]
- BAIR/CPAR/BDD Internal Weekly Seminar
- DeepDrive Berkeley [[Link](http://bdd.berkeley.edu/)]
- Mobile Robotics Group Oxford 


## 4 Autonomous Driving Datasets

#### comma.ai Dataset 

> 7:15h of highway driving
> [[Link](https://github.com/commaai/research)]

#### KITTI Vision Benchmark Suite 

> [[Link](http://www.cvlibs.net/datasets/kitti/)]

- Stereo: 200 training, 200 testing
- Optical Flow: 200 training, 200 testing
- Scene Flow: 200 training, 200 testing
- Visual Odometry: 22 videos of 40km
- Object Detection: 7500 training, 7500 testing
- Object Tracking: 21 training, 29 testing
- Roag segmentation: 289 training, 290 testing

#### Cityscapes Dataset 

- 5000 annotated images with fine annotations
- 20000 annotated images with coarse annotations 

#### DARPA Urban Grand Challenge Dataset 

> [[Link](http://grandchallenge.mit.edu)]

#### Udacity Open Dataset 

> 223GB (70 min driving in Mountain View) 
> [[Link](https://github.com/udacity/self-driving-car)]

#### DeepDrive.io Dataset 

> 80GB (42h of driving simulated in GTAV) 
> [[Link](https://deepdrive.io)]

#### Playing for Data: Ground Truth from Computer Games 

> (24966 densely labeled frames) 
> [[Link](http://download.visinf.tu-darmstadt.de/data/from_games/index.html)]

#### Oxford Robot Car Dataset

> Huge dataset collected over 1 year with 1000km driving 
> [[Link](http://robotcar-dataset.robots.ox.ac.uk/)]


## 5 Simulation Environment and Data Generation

- Driverseat (Crowdsourcing)
- GTA V 
- TORCS


### Websites

- deepdrive.io
- dvbuntu.github.io/wisdom

## 6 Projects

### Deep RL projects

- rllab: Benchmarking Deep RL [[Code](https://github.com/rllab/rllab)]
- keras-rl: Deep RL library in Keras [[Code](https://github.com/matthiasplappert/keras-rl)]
- implementation of RL algorithms for Richard Sutton's book [[Code](https://github.com/dennybritz/reinforcement-learning)]

### AD projects

### More Links

- Autonomous Vision Group [[Link](http://www,cvlibs.net/links.php)]
