#!/bin/bash

if [[ ! -z "$1" ]] ; then
    if [ "$1" == true ]
    then
        echo 'compiling.'
        rm -r dist
        rm -r build
        touch ~/.pypirc
        python3 setup.py sdist bdist_wheel
        echo 'compiling complete!'
    fi
fi

if [[ ! -z "$2" ]]
then
    echo 'test begin.'
    pip install $2
    echo 'test complete!'
fi
