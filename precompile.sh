#!/bin/bash

if [[ ! -z "$1" ]] ; then
    if [ "$1" == true ] ; then
        echo 'compiling.'
        rm -r dist
        rm -r build
        rm -r info
        touch ~/.pypirc
        python3 setup.py sdist bdist_wheel
        echo 'compiling complete!'
    fi
fi

if [[ ! -z "$2" ]] ; then
    echo 'test begin.'
    pip install $2
    echo 'test complete!'
fi

if [[ ! -z "$3" ]] ; then
    if [ "$3" == true ] ; then
        rm -r .ipynb_checkpoints
        rm -r classix/.ipynb_checkpoints
        mv 'ClassixClustering.egg-info' info
    fi
fi
