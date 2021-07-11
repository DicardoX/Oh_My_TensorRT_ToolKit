# Oh_My_TensorRT_ToolKit

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

## Table of Contents

- [Background](#background)
- [Installation](#installation)
- [Usage](#usage)
	- [Generator](#generator)
- [Examples](#examples)
- [Related Efforts](#related-efforts)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
- [License](#license)

--------

## Background

###### Introduction

*Oh_My_TensorRT_ToolKit* is **a wrapped toolkit for GPU performance test using various NN models based on [TensorRT official source code](https://github.com/NVIDIA/TensorRT).** 

> I develop this toolkit only for providing the assistance of GPU performance test when we use the official TensorRT tools. Besides, this toolkit could be highly customed in different versions of TensorRT, please refer to the [Contributing](#contributing) part.

 *Oh_My_TensorRT_ToolKit* needs serveral procedures to established, which could be found in [Usage](#usage) part. We implement this toolkit by basicly using the [./trtexec](https://github.com/NVIDIA/TensorRT/tree/master/samples/opensource/trtexec), a TensorRT Command-Line Wrapper provided by NVIDIA TensorRT official samples, in our scripts.

The **workload** of *Oh_My_TensorRT_ToolKit* is:

1. **Generate Onnx models from our Pytorch / Tensorflow models**. We provide both *Pytorch* and *Tensorflow* examples in [Examples](#examples) part.
2. **Generate dynamic engines from the Onnx models by using ./trtexec tool**.
3. **Perform inference tests with the engines in different MIG devices and batch sizes**.
4. **Format output the throughput / latency statistical results**.

As for the generation of onnx model, we provide both *Pytorch* and *Tensorflow* examples in [Examples](#examples) part.

###### GPU Environment

 *Oh_My_TensorRT_ToolKit* should be used on GPU environment, and our standard GPU device and drivers are:

- **GPU Device**: **A100-PCIE-40GB with MIG mechanism** (temporarily 3 MIG devices)

```shell
GPU 0: A100-PCIE-40GB (UUID: GPU-9de3d0e8-33f5-10dc-0c79-2c88a7ab0a23)
  MIG 4g.20gb Device 0: (UUID: MIG-GPU-9de3d0e8-33f5-10dc-0c79-2c88a7ab0a23/2/0)
  MIG 2g.10gb Device 1: (UUID: MIG-GPU-9de3d0e8-33f5-10dc-0c79-2c88a7ab0a23/3/0)
  MIG 1g.5gb Device 2: (UUID: MIG-GPU-9de3d0e8-33f5-10dc-0c79-2c88a7ab0a23/9/0)
```

​	More introductions on MIG mechanism could be found in [Basic on MIG Tutorial](https://github.com/DicardoX/Oh_My_TensorRT_ToolKit/blob/main/docs/Basic_on_MIG.md).

- **Drivers**:
    1. NVIDIA-SMI 460.80, Driver Version: 460.80
    2. **CUDA Version**: 11.2
    3. **TensorRT Version**: 8.0.1.6
    4. **Torch Version**: 1.9.0+cu111 (Customed configurations could be found in [Pytorch Official](https://pytorch.org/get-started/locally/))
    5. **Torchvision Version**: 0.10.0+cu111
    6. **Pytorch-pretrained-bert Version**: 0.6.2 (This is for the usage of bert model)
    7. **Onnx Version**: 1.9.0
    8. **Netron Version**: 5.0.0
    9. **Onnxruntime Version**: 1.2.0
    10. **Tensorflow Version**: 2.5.0
    11. **H5py Version**: 3.1.0

------------

## Installation



This project uses [node](http://nodejs.org) and [npm](https://npmjs.com). Go check them out if you don't have them locally installed.

```sh
$ npm install --global standard-readme-spec
```

## Usage

This is only a documentation package. You can print out [spec.md](spec.md) to your console:

```sh
$ standard-readme-spec
# Prints out the standard-readme spec
```

### Generator

To use the generator, look at [generator-standard-readme](https://github.com/RichardLitt/generator-standard-readme). There is a global executable to run the generator in that package, aliased as `standard-readme`.

## Examples

To see how the specification has been applied, see the [example-readmes](example-readmes/).

## Related Efforts

- [Art of Readme](https://github.com/noffle/art-of-readme) - 💌 Learn the art of writing quality READMEs.
- [open-source-template](https://github.com/davidbgk/open-source-template/) - A README template to encourage open-source contributions.

## Maintainers

[@RichardLitt](https://github.com/RichardLitt).

## Contributing

Feel free to dive in! [Open an issue](https://github.com/RichardLitt/standard-readme/issues/new) or submit PRs.

Standard Readme follows the [Contributor Covenant](http://contributor-covenant.org/version/1/3/0/) Code of Conduct.

### Contributors

This project exists thanks to all the people who contribute. 
<a href="https://github.com/RichardLitt/standard-readme/graphs/contributors"><img src="https://opencollective.com/standard-readme/contributors.svg?width=890&button=false" /></a>


## License

[MIT](LICENSE) © Richard Littauer

A wrapped toolkit for GPU performance test using various NN models based on TensorRT official source code.
