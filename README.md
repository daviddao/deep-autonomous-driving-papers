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

## 1. Planning for Deep Autonomous Driving

> There are currently three paradigms for vision-based autonomous driving systems: 

> 1. *Mediated Perception* parses every scene into structured data and derives decision from that
> 2. *Behaviour Reflex* maps directly from input image to driving control using deep learning 
> 3. *Direct Perception* maps input to a small number of perception indicators and make a driving decision

---------------------------------------

### 1.1 Mediated Perception 

**[1]** B Huval, T Wang, S Tandon, J Kiske, W Song, J Pazhayampallil, M Andriluka, P Rajpurkar, T Migimatsu, R Cheng-Yue, F Mujica, A Coates, AY Ng.  **An Empirical Evaluation of Deep Learning on Highway Driving** (2015)

> Uses Overfeat for real-time vehicle and lane detection. Due to detection ambiguity, the authors modified Overfeat to first predict a object mask and then used a bounding box regression for the final object detection.

> - Architecture: Overfeat
> - Input: 640x480 input 
> - Output: Object mask

**[2]** A Provodin, L Torabi, B Flepp, Y LeCun, M Sergio, LD Jackel, U Muller, J Zbontar. **Fast Incremental Learning for Off-Road Robot Navigation** (2016)

> Uses transfer learning to cope with the problem of missing data in AD. Shows that a pretrained CNN on ImageNet can extract good features for AD.

> - Architecture: 3-layer CNN
> - Input: 59x59 image patches around pixels
> - Output: Each pixel as "drivable" or "obstacle" 

> Just visual but no empirical results.

**[3]** Z Zhang, S Fidler, R Urtasun. **Instance-Level Segmentation for Autonomous Driving with Deep Densely Connected MRFs** (2016)

**[4]** P Ondruska, I Posner **Deep Tracking: Seeing Beyond Seeing Using Recurrent Neural Networks** (AAAI 2016) [[code](https://github.com/pondruska/DeepTracking)] :star: 

> Introduces the DeepTracking framework. A RNN can be trained to track objects (even when occluded) only using the raw sensory input. F1 score is 0.6 for 1 sec prediction of future occupancy.

> - Architecture: 3-layer CNN-GRU RNN
> - Input: Raw occupancy map
> - Output: Object map

> The authors use two tricks: Raw sensor input contains weak supervision and they predict future data points (instead present) to model movement.

**[5]** J Dequaire, D Rao, P Ondruska, DZ Wang, I Posner. **Deep Tracking on the Move: Learning to Track the World from a Moving Vehicle using Recurrent Neural Networks** (2016) :star:

> End-to-end approach for tracking objects in a moving car using the DeepTracking framework. It can track occluded objects using an RNN and account for the movement of the vehicle by adding spatial invariance. 0.7 F1 score for 1 sec prediction time. STM contributes up to 0.07 F1 score.

> - Architecture: 3-layer CNN-GRU RNN with STN 
> - Input: Raw occupancy map
> - Output: Object map 

**[6]** P Ondruska, J Dequaire, DZ Wang, I Posner. **End-to-End Tracking and Semantic Segmentation Using Recurrent Neural Networks** (2016)

> DeepTracking trained with unsupervised and (weak) supervised data to perform tracking and semantic segmentation. Network learns implicit tracking and world states which can be used for segmentation using transfer learning. Improved F1 score by 0.05 from original paper.

> - Architecture: 3-layer CNN-GRU RNN with bias and dilated convolutions
> - Input: Raw occupancy map, semantic segmentation map
> - Output: Object map with semantic segmentation

> The author extends the DeepTracking architecture using dilated convolutions to track objects of different scales, dynamic memory (LSTM or GRU) for information caching, static memory which helps to store place-specific information. 

**[7]** D Barnes, W Maddern, I Posner **Find Your Own Way: Weakly-Supervised Segmentation of Path Proposals for Urban Autonomy** (2016)

> Creates more path segmentation training data by combining future frames and sensor information. Uses a SegNet to train and predict paths using this additional data. Evaluates on KITTI and Oxford Car dataset.

> - Architecture: SegNet
> - Input: 640x256 image
> - Output: Segmented image with path, unknown, obstacle

> Around 85-93% accuracy on Oxford, improving previous SegNet results on KITTI by up to 20%.

---------------------------------------

### 1.2 Behaviour Reflex (Supervised End-to-End Learning)

> General Design: Input -> NN -> Control

**[1]** DA Pomerleau. **ALVINN, An Autonomous Land Vehicle in a Neural Network** (1988) :star:

> First seminal work of a neural network for autonomous driving, trained on simulated data and achieving 90% direction prediction accuracy (left or right) on road following.

> - Architecture: 2-layer FCN 
> - Input: 30x32 image, 8x32 laser range (Simulated)
> - Output: Direction 

> During testing, intensity output is recirculated to the input unit, which improves accuracy.

**[2]** Y LeCun, U Mueller, J Ben, E, Cosatto, B Flepp. **Off-Road Obstacle Avoidance through End-to-End Learning** (2005) :star:

> First use of a CNN for autonomous driving, introduces DAVE an off-road autonomous vehicle trained end-to-end to avoid obstacles. 

> - Architecture: 6-layer CNN 
> - Input: left/right pair of 149x58 image 
> - Output: Steering angle 

> Mean distance between crashes is 20m 

**[3]** R Hadsell, P Sermanet, J Ben, A Erkan, M Scoffier, K Kavukcuoglu, U Mueller, Y LeCun. **Learning Long-Range Vision for Autonomous Off-Road Driving** (2009)

> Building upon DAVE, it describes DARPA's LAGR.


**[4]** J Koutnik, G Cuccu, J Schmidhuber, FJ Gomez. **Evolving large-scale neural networks for vision-based torcs** (2013)

> The authors use neuroevolution with a compressed weight matrix representation instead of backpropagation to learn a set of weights for TORCS (1 million).

> - Architecture: Neuroevolved RNN
> - Input: Image, 25sec driving frame
> - Output: Driving behaviour, new set of genome

> Defined a custom fitness function, ran the experiments with 200 generations.
 
**[5]** M Bojarski, DD Testa, D Dworakowski, B Firner, B Flepp, P Goyal, LD Jackel, M Monfort, U Muller, J Zhang, X Zhang, J Zhao, K Zieba. **End to End Learning for Self-Driving Cars** (2016) :star:

> DAVE-2 trains a CNN end-to-end and has a 98% autonomy rate. The authors show that just by using steering as sparse signal and 100h driving, the CNN is able to learn useful features (ie outline of the road).
> The authors evaluated driving on highway and roads under different weather circumstances, without lane switches and road changes.

> - Architecture: 9-layer CNN (chosen empirically)
> - Input: 200x66 images sampled from Video (10FPS)
> - Output: Inverse turning radius

> Represents steering command as *1/r*, where *r* is the turning radius in meters. 
> Train the car with negative examples using shifted cameras.
> Ranking after autonomous score: Each intervention is a penalty of 6 seconds.

---------------------------------------

### 1.3 Direct Perception

> General Design: Input -> NN -> Indicators -> Controller -> Control

**[1]** C Chen, A Seff, A Kornhauser, J Xiao. **DeepDriving: Learning Affordance for Direct Perception in Autonomous Driving** (ICCV2015) :star:

> Describes well pro and cons of mediated and end-to-end learning and introduces another approach: direct perception. Uses a CNN to learn affordance values for a controller. Focuses on Highway Driving with multiple lanes.

> - Architecture: AlexNet  
> - Input: 280x210 image (simulated)
> - Output: 13 affordance indicators

> Evaluated on KITTI and TORCS. Tested visually on Smartphone video. Speed does not exceed 72km/h, reliable car perception within 30m. Use close and long range ConvNet for KITTI (input cropped 497 x 150). Comparable performance to state-of-the-art car distance estimation (6m mean error).

---------------------------------------

### 1.4 Reinforcement Learning

> Uses policy to estimate the best action in a given state

**[1]** J Zhang, K Cho. **Query-Efficient Imitation Learning for End-to-End Autonomous Driving** (2016)

> A human driver (reference policy) cannot cover all situations in data. This paper introduces imitation learning for AD, where a CNN learns a primary policy and together with the reference policy iterate to generate more data. Approach based on DAgger
> A safety policy, estimated by an additional FCN, predicts, if it is safe for a NN to drive. Evaluated on TORCS only.

> - Architecture: 6-layer CNN, 2 layer FCN 
> - Input: 160x72 image (simulated in TORCS), Conv5
> - Output: Steering angle, safe/unsafe



**[2]** M Wulfmeier, DZ Wang, I Posner. **Watch This: Scalable Cost-Function Learning for Path Planning in Urban Environments** (IROS16 Best Student Paper)

> Extends Maximum Entropy Inverse RL by using a multi-scale F-CNN architecture trained on 25000 trajectories. The trained network is robust towards sensor miscalibration and outperforms hand-designed features. Evaluated on real-world data. 20% FNR for 0% FPR trajectory prediction (compared to hand-tuned state-of-the art with 50% FPR)

> - Architecture: 9-layer Fully-CNN with Max Pool and Multi-Scale Architecture 
> - Input: 15m long trajectories on a 2D based 50x50m static map with 0.25m resolution (LIDAR)
> - Output: Costmap


---------------------------------------

## 2 Theory


**[1]** M Wulfmeier, P Ondruska, I Posner. **Maximum Entropy Deep Inverse Reinforcement Learning** (NIPS RL Workshop 2015) 

> Maximum Entropy IRL models suboptimal samples (i.e driver trajectories) by assign each sample a probability proportional to its expected reward. The authors then use a CNN to predict costs for large state spaces and complex reward structures, achieving state-of-the-art.

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

### 3 Important Researchers in Deep AD

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

**[1]** **Comma.ai Dataset** [[Link](https://github.com/commaai/research)]

> 7:15h of highway driving

**[2]** **KITTI Vision Benchmark Suite**  [[Link](http://www.cvlibs.net/datasets/kitti/)]

> - Stereo: 200 training, 200 testing
> - Optical Flow: 200 training, 200 testing
> - Scene Flow: 200 training, 200 testing
> - Visual Odometry: 22 videos of 40km
> - Object Detection: 7500 training, 7500 testing
> - Object Tracking: 21 training, 29 testing
> - Roag segmentation: 289 training, 290 testing

**[3]** **Cityscapes Dataset** [[Link](https://www.cityscapes-dataset.com/)]

> 5000 annotated images with fine annotations
> 20000 annotated images with coarse annotations 

**[4]** **DARPA Urban Grand Challenge Dataset** [[Link](http://grandchallenge.mit.edu)]

> 200 GB (Log files of sensors and camera logs)

**[5]** **Udacity Open Dataset** [[Link](https://github.com/udacity/self-driving-car)]

> 223GB (70 min driving in Mountain View) 

**[6]** **DeepDrive.io Dataset** [[Link](https://deepdrive.io)]

> 80GB (42h of driving simulated in GTAV) 

**[7]** **Playing for Data: Ground Truth from Computer Games**  [[Link](http://download.visinf.tu-darmstadt.de/data/from_games/index.html)]

> (24966 densely labeled frames) 

**[8]** **Oxford Robot Car Dataset** [[Link](http://robotcar-dataset.robots.ox.ac.uk/)]

> Huge dataset collected over 1 year with 1000km driving 

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

### More Links

- Autonomous Vision Group [[Link](http://www,cvlibs.net/links.php)]
