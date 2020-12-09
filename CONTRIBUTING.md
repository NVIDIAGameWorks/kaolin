# Contributing guidelines

## Pull Request Checklist

Before sending your pull requests, make sure you followed this list.

1) Read these Guidelines in full.

2) Please take a look at the LICENSE (it's Apache 2.0).

3) Make sure you sign your commits. E.g. use ``git commit -s`` when commiting.

4) Check your changes are consistent with the [Standards and Coding Style](CONTRIBUTING.md#standards-and-coding-style).

5) Make sure all unittests finish successfully before sending PR.

6) Send your Pull Request to the `master` branch

## How to become a contributor and submit your own code

### Signing your commits
Before we can take your patches we need to take care of legal concerns.

Please sign each commits using ``git commit -s``.
In case you forgot to sign previous commits you can amend previous commits using:
* ``git commit -s --amend`` for the last commit.
* ``git rebase --signoff`` for all the commits of your pull request.

### Contributing code
Contributions are welcome!
You can send us pull requests to help improve Kaolin, if you are just getting started, Gitlab has a [how to](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html).

Kaolin team members will be assigned to review your pull requests. Once they your change passes the review and the continuous integration checks, a Kaolin member will approve and merge them to the repository.

If you want to contribute, [Gitlab issues](https://gitlab-master.nvidia.com/Toronto_DL_Lab/kaolin-reformat/-/issues) are a good starting point, especially the ones with the label [good first issue](https://gitlab-master.nvidia.com/Toronto_DL_Lab/kaolin-reformat/-/issues?scope=all&utf8=%E2%9C%93&state=opened&label_name[]=good%20first%20issue). If you started working on a issue, leave a comment so other people know that you're working on it, you can also coordinate with others on the issue comment threads.

### Standards and Coding Style
#### General guidelines
* New features must include unit tests which help guarantee correctness in the present and future.
* API changes should be minimal and backward compatible. Any changes that break backward compatibility should be carefully considered and tracked so that they can be included in the release notes.
* New features may not accepted if the cost of maintenance is too high in comparison of its benefit, they may also be integrated to contrib subfolders for minimal support and maintenance before eventually being integrated to the core.

#### License
Include a license at the top of new files.

* [C/C++/CUDA example](example_license.cpp)
* [Python example](examples_license.py)

#### Code organization
* [kaolin](kaolin/) - The Core of Kaolin library, everything that is not in [csrc](kaolin/csrc) is a Python module.
  * [csrc](kaolin/csrc/) - Directory for all the C++ / CUDA implementations of custom ops.
    The gpu ops parts will be under the subdirectory [csrc/cuda](kaolin/csrc/cuda)
    while the cpu parts will be under the subdirectory [csrc/cpu](kaolin/csrc/cpu).
  * [io](kaolin/io/) - Module of all the I/O features of Kaolin, such a importing and exporting 3D models.
  * [metrics](kaolin/metrics) - Module of all the metrics that can be used as differentiable loss or distance.
  * [ops](kaolin/ops/) - Module of all the core operations of kaolin on different 3D representations.
  * [render](kaolin/render/) - Module of all the differentiable renderers modules and advanced implementations.
  * [utils](kaolin/utils/) - Module of all the utility features for debugging and testing.
  * [visualize](kaolin/visualize/) - Module of all the visualization modules.
* [examples](examples/) - Examples of Kaolin usage
* [tests](tests/) - Tests for all Kaolin

#### C++ coding style
We follow [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)

It is enforced using [Clang-Tidy](https://clang.llvm.org/extra/clang-tidy/index.html)

#### Python coding style
We follow [PEP8 Style Guide](https://www.python.org/dev/peps/pep-0008/) with some exceptions listed in [flake8 config file](https://gitlab-master.nvidia.com/Toronto_DL_Lab/kaolin-reformat/.flake8) and generally follow PyTorch naming conventions.

It is enforced using [flake8](https://pypi.org/project/flake8/), with [flake8-bugbear](https://pypi.org/project/flake8-bugbear/), [flake8-comprehensions](https://pypi.org/project/flake8-comprehensions/), [flake8-mypy](https://pypi.org/project/flake8-mypy/) and [flake8-pyi](https://pypi.org/project/flake8-pyi/)

to run flake8 execute ``flake8 --config=.flake8 .`` from the [root of kaolin](https://gitlab-master.nvidia.com/Toronto_DL_Lab/kaolin-reformat)

On top of that we use prefixes (``packed\_``, ``padded\_``) to indicate that a module / op is specific to a layout, an , all ops of the same purpose for different layouts should be in the same file.

[tests/python/kaolin/](tests/python/kaolin) should follows the same directory structure of [kaolin/](kaolin/). E.g. each module kaolin/path/to/mymodule.py should have a corresponding tests/python/kaolin/path/to/test\_mymodule.py.

#### Tests
We are applying [pytest](https://docs.pytest.org/en/latest/) on [tests/python directory](tests/python/), with [pytest-cov](https://pytest-cov.readthedocs.io/en/latest/), which should follows the directory structure of [kaolin](kaolin/).

to run the tests execute ``pytest --cov=kaolin/ tests/`` from the [root of kaolin](https://gitlab-master.nvidia.com/Toronto_DL_Lab/kaolin-reformat)

#### Documentation
Contributors are encouraged to verify the generated documentation before each pull request.

To build your own documentation, follow the [guide](docs/README.md).
