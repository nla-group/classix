MATLAB's users
=======================

CLASSIX's provide easy-to-use Maltab package `classix-matlab <https://github.com/nla-group/classix-matlab>`_. Besides, you can easily setup call Python's CLASSIX in Matlab.

We will provide a simple demostration for Matlab's users to call CLASSIX from Matlab.


.. admonition:: Note

    This example is provided by Mike Croucher, one can access the original scripts of live code format (.mlx) and jupyter notebook file (.ipynb) in  `CLASSIX's GitHub repository <https://github.com/nla-group/classix/tree/master/matlab>`_.
    For more details of using Python's package in Matlab and MATLAB kernel for Jupyter released, feel free to visit `Mike's blog <https://blogs.mathworks.com/matlab/2023/01/30/official-mathworks-matlab-kernel-for-jupyter-released/>`_ for answers.


++++++++++++++++++
Enviroment setup
++++++++++++++++++

First, ensure you reference Python (cf. `https://www.python.org <https://www.python.org/>`_) and Python's CLASSIX is properly presented (cf. `Installation guide <https://classix.readthedocs.io/en/latest/quickstart.html>`_)

Then, you can check your Python enviroment in your Matlab using:

.. code:: matlab

    pe = pyenv()

In Windows platform, it might show the below information:

.. parsed-literal::

    pe = 
    PythonEnvironment with properties:

            Version: "3.11"
        Executable: "c:\Users\...\AppData\Local\Programs\Python\Python311\python.exe"
            Library: "c:\Users\...\AppData\Local\Programs\Python\Python311\python311.dll"
                Home: "c:\Users\...\AppData\Local\Programs\Python\Python311"
            Status: NotLoaded
        ExecutionMode: InProcess



Otherwise, you can setup enviroment in your Matlab via below command:

.. code:: matlab

    pe = pyenv(Version="C:\Users\cclcq\AppData\Local\Programs\Python\Python311\python.exe")


All Python commands must be prefixed with py. So, to compute the Python command math.sqrt(42) we simply do:

.. code:: matlab

    py.math.sqrt(42)

++++++++++++++++++
Clustering analysis
++++++++++++++++++

After ensuring everying is presented, we can performing a basic clustering analysis on MATLAB data using CLASSIX as below:

Let's start by generating and plotting some data using MATLAB.

.. code:: matlab

    mu1 = [2 2];          % Mean of the 1st cluster
    sigma1 = [2 0; 0 1];  % Covariance of the 1st cluster
    mu2 = [-4 -3];        % Mean of the 2nd cluster
    sigma2 = [1 0; 0 1];  % Covariance of the 2nd cluster
    r1 = mvnrnd(mu1,sigma1,100);
    r2 = mvnrnd(mu2,sigma2,100);
    X = [r1; r2];

Calling CLASSIX is straightforward. We don't even need to convert the MATLAB array X to a Numpy array as it's all done automatically.

.. code:: matlab

    rng('default')        % For reproducibility

    plot(X(:,1),X(:,2),"*",MarkerSize=5);
    clx = py.classix.CLASSIX(radius=0.3, verbose=0);
    clx = clx.fit(X);
    clx.explain(plot=false);



The cluster labels of each data point are available in clx.labels_. This is a Numpy array:

.. code:: matlab

    class(clx.labels_)

but no conversion is required when using this in the MATLAB scatter command:

.. code:: matlab

    scatter(X(:,1),X(:,2),10,clx.labels_,"filled");

+++++++++++++++++++++++++++++++++++++++
Explainability and plotting
+++++++++++++++++++++++++++++++++++++++

A key feature of CLASSIX is that it can provide textual explanations of the computed clustering results, making it a fully explainable clustering algorithm. The CLASSIX ``explain()`` method can also produce plots, but you may receive an error message when attempting to do this from MATLAB:

.. code:: matlab

    clx.explain(plot=true)

``explain()`` method requires MATLAB TCL installed, this is explained on MATLAB Answers at `Why am I not able to call python Tkinter in MATLAB? - MATLAB Answers - MATLAB Central (mathworks.com) <https://uk.mathworks.com/matlabcentral/answers/808595-why-am-i-not-able-to-call-python-tkinter-in-matlab?s_tid=srchtitle>`_. We need to provide paths to TCL.

To setup the enviroment, use:

.. code:: matlab

    setenv('TCL_LIBRARY', 'C:\Program Files\Python311\tcl\tcl8.6')
    setenv('TK_LIBRARY', 'C:\Program Files\Python311\tcl\tk8.6')
    clx.explain(plot=true)


.. admonition:: Note

    One need to find the correct paths on your machine for MATLAB TCL.