### Basic on MIG

[【深入GPU硬件架构及运行机制】](https://www.cnblogs.com/timlly/p/11471507.html)

[【查看gpu使用率 nvidia_如何玩转安培架构的 MIG (多实例GPU) 及其应用案例分享】](https://blog.csdn.net/weixin_39626181/article/details/111123999)

#### 1.1 GPU实例的设定

- **在A100 GPU下，最多被划分为7个GPU实例，7个流处理器 (compute)，8个独立的内存，分别对应图二中各层级的部分**。
- 下图展示了不同大小GPU实例的相关资源占有情况。例如，MIG 2g.10gb尺寸的GPU实例最多可分配3份（此时会有浪费），具有2份流处理器和2份内存。**在使用时，我们可以同时使用多类GPU实例的组合，以灵活分配实例的大小**。

<img src="./cut/截屏2021-07-01 下午3.18.11.png" alt="avatar" style="zoom:40%;" />

​																					图一 GPU实例的设定

- 举例来说，我们可以使用一个 4g.20gb 实例、一个 2g.10gb 实例，以及一个 1g.5gb 实例。图二展示了我们使用的 GPU 实例。这里需要注意的是，虽然我们使用了所有的流处理器，但只有用到 7 份的内存，有一份内存会浪费掉。

<img src="./cut/截屏2021-07-01 下午3.19.17.png" alt="avatar" style="zoom:40%;" />

​																					图二 GPU实例的组合

----------

#### 1.2 MIG实例的管理

- **管理者可以通过赋予使用者访问特定档案的权限来限制使用者对 MIG 实例的使用权限**，该档案放在 `/proc/driver/nvidia/capabilities` 目录下。
- 对于 MIG 的管理，可以**通过 `mig/config` 和 `mig/monitor` 来控制权限**。
    - `mig/config` 一般只有 root 权限有访问的权限。拥有访问这个目录权限的使用者能够管理 MIG 的实例，例如实例的创建和删除
    - `mig/monitor` 的权限则能让使用者看到整个 GPU 的信息，例如目前 GPU 内存的使用量、是否开启 MIG 模式等等
- 除了上述两个管理和监控 MIG 工具的权限之外，MIG 也提供**针对特定 GPU 实例或是计算实例的使用权限**。例如若**要让使用者能够只能使用图三当中的 gpu0 中 gi0 GPU实例下的ci0计算实例**，则赋予该使用者访问 `gpu0/gi0/access` 和 `gpu0/gi0/ci0/access` 这两个档案的访问权限，并且移除该使用者访问其他实例的权限。

<img src="./cut/截屏2021-07-01 下午3.40.13.png" alt="avatar" style="zoom:40%;" />

​																			图三 MIG管理目录的文件结构

---------

#### 1.3 MIG的使用

###### 1.3.1 MIG模式的开启

- 注意，开启MIG需要root权限，使用如下指令：

    ```shell
    sudo nvidia-smi mig 1
    ```

- **开启 MIG 之后，在还没有建立 GPU 实例和计算实例之前，是不能使用GPU的**。如果使用者在这个情况下直接去执行程序，会返回找不到相关装置的错误。

- 成功开启后，`nvidia-smi` 界面发生变化，增加了一栏对MIG的说明：

    - GI：GPU Instance
    - CI：Compute Instance

<img src="./cut/截屏2021-07-01 下午4.07.55.png" alt="avatar" style="zoom:40%;" />

###### 1.3.2 MIG下GPU实例与计算实例的创建

- 确认能够创建的GPU实例：

    ```shell
     sudo nvidia-smi mig -lgip
    ```

- 创建特定ID的GPU实例：

    ```shell
    sudo nvidia-smi mig -cgi [ID]
    ```

    注意，由于需要使用root权限，本部分不做详细介绍，具体参见 [Ref](https://blog.csdn.net/weixin_39626181/article/details/111123999) 中的**创建 GPU 实例与计算实例**模块。

------

#### 1.4 在指定的计算实例上运行程序

- 创建完计算实例之后，我们要如何在特定的实例上面执行我们的程序呢？目前，我们要在指定的 GPU 上面执行程序，是通过 **“CUDA_VISIBLE_DEVICES” 这一环境变量**。而 CUDA 11 在这个环境变数上进行了扩充，除了能够指定第几个 GPU 之外，也能通过**计算实例的 UUID** 来直接指定使用哪个计算实例。
- 通过 `nvidia-smi -L` 来得到所有计算实例的UUID。

<img src="./cut/截屏2021-07-01 下午6.29.03.png" alt="avatar" style="zoom:40%;" />

- 在得到计算实例的 UUID 之后，我们便可以通过 `CUDA_VISIBLE_DEVICES=[UUID]` 这个环境变量来使得程序在指定的计算实例上运行。
- 吞吐量和latency如何定义？
    - Latency：每个batch执行的时间，注意用在**模型预测**而不是模型训练！
    - Throughput = (1000 / latency) * batch_size
