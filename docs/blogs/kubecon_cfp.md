# ElasticDL: Kubernetes-Native Distributed Deep Learning with Elastic Scheduling

## Description

In addition to online services, users have been running distributed AI jobs on
Kubernetes. Related projects add Kubernetes operators to start distributed deep
learning programs calling TensorFlow or PyTorch. In these solutions, Kubernetes
plays the role to launch pods and restart preempted ones. However, such retrials
often fail due to the same reason that caused the perception -- the lack of
resources. If couldn't maintain the constant number of worker pods, jobs fail.
Maintaining the constant number, it becomes gang scheduling. Either case leads
to low utilization of clusters.

ElasticDL changes the paradox by using the statistical properties of distributed
learning theory and making jobs tolerable to varying numbers of workers. Taking
it as the basis, ElasticDL realizes elastic scheduling by introducing a master
pod per job, to replace the Kubernetes operator per cluster. It makes full use
of residual resources and improve the utilization significantly.

## Benefits to the Ecosystem

ElasticDL boosts the cluster utilization up to 90%, on on-premise clusters at Ant
Financial and on Google Cloud, as it makes full use of residual resources to run
deep learning jobs with elastic scheduling. Moreover, it enables the running of
deep learning jobs in lower priority than online services on the same cluster.
It senses and uses resource left by online services.

The master is important to elastic scheduling, it takes three roles.

1. The master dynamically partitions the training data so to decouple the number
of partitions and that of the workers.
2. The master also works with Kubernetes to watch the cluster utilization and a
good chance to restart failed workers. Before the chance, the master always
leverages living workers.
3. The master starts and monitors parameter servers when training large models
using the asynchronous SGD algorithm, and cooperate workers to implement a
Kubernetes-native fault-tolerable AllReduce operation for the synchronous SGD
counterpart.

Deep learning researchers like ElasticDL as it reduces the pending time of each
job as it makes full use of residual resources to run as many concurrent
experiments as possible. Deep learning jobs depend on many parameters,
including the optimizer, activator, and cost, hyperparameters. Users are eager
to see the status of the first few iterations of the learning job so to ensure
the configuration is mathematically correct. ElasticDL provides a cure. Using
the residual resource, ElasticDL also shortens the total time to run a batch
of training jobs.

ElasticDL provides an easy-to-use interface. Users define models using
TensorFlow 2.x API just like filling the map and reduce functions required by
the MapReduce framework, without the need to consider anything about
distributed programming. The interface allows users to test their models
locally and run big data using ElasticDL without changing their source code.

## Open Source Projects

[ElasticDL](https://github.com/sql-machine-learning/elasticdl)

## Speakers
