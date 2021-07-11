# Oh_My_TensorRT_ToolKit

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
	- [Generator](#generator)
- [Example Readmes](#example-readmes)
- [Related Efforts](#related-efforts)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
- [License](#license)

--------

## Background

###### Introduction

*Oh_My_TensorRT_ToolKit* is **a wrapped toolkit for GPU performance test using various NN models based on [TensorRT official source code](https://github.com/NVIDIA/TensorRT).** 

> I develop this toolkit only for providing the assistance of GPU performance test when we use the official TensorRT tools. Besides, this toolkit could be highly customed in different versions of TensorRT, please refer to the [Contributing](#contributing) part.

 *Oh_My_TensorRT_ToolKit* needs serveral procedures to established, which could be found in [Usage](#usage) part.

###### Environment

 *Oh_My_TensorRT_ToolKit* should be used on GPU environment, and our standard GPU device and drivers are:

- **GPU Device**: **A100-PCIE-40GB with MIG mechanism** (temporarily 3 MIG devices)

```shell
GPU 0: A100-PCIE-40GB (UUID: GPU-9de3d0e8-33f5-10dc-0c79-2c88a7ab0a23)
  MIG 4g.20gb Device 0: (UUID: MIG-GPU-9de3d0e8-33f5-10dc-0c79-2c88a7ab0a23/2/0)
  MIG 2g.10gb Device 1: (UUID: MIG-GPU-9de3d0e8-33f5-10dc-0c79-2c88a7ab0a23/3/0)
  MIG 1g.5gb Device 2: (UUID: MIG-GPU-9de3d0e8-33f5-10dc-0c79-2c88a7ab0a23/9/0)
```

​	More introductions on MIG mechanism could be found in []()

- d





Writing READMEs is way too hard, and keeping them maintained is difficult. By offloading this process - making writing easier, making editing easier, making it clear whether or not an edit is up to spec or not - you can spend less time worrying about whether or not your initial documentation is good, and spend more time writing and using code.

By having a standard, users can spend less time searching for the information they want. They can also build tools to gather search terms from descriptions, to automatically run example code, to check licensing, and so on.

The goals for this repository are:

1. A well defined **specification**. This can be found in the [Spec document](spec.md). It is a constant work in progress; please open issues to discuss changes.
2. **An example README**. This Readme is fully standard-readme compliant, and there are more examples in the `example-readmes` folder.
3. A **linter** that can be used to look at errors in a given Readme. Please refer to the [tracking issue](https://github.com/RichardLitt/standard-readme/issues/5).
4. A **generator** that can be used to quickly scaffold out new READMEs. See [generator-standard-readme](https://github.com/RichardLitt/generator-standard-readme).
5. A **compliant badge** for users. See [the badge](#badge).

## Install

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

## Example Readmes

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
