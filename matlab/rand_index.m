function ri = rand_index(p1, p2, varargin)
%RAND_INDEX Computes the rand index between two partitions.
%   RAND_INDEX(p1, p2) computes the rand index between partitions p1 and
%   p2. Both p1 and p2 must be specified as N-by-1 or 1-by-N vectors in
%   which each elements is an integer indicating which cluster the point
%   belongs to.
%
%   RAND_INDEX(p1, p2, 'adjusted') computes the adjusted rand index
%   between partitions p1 and p2. The adjustment accounts for chance
%   correlation.
%
% This code is by Chris McComb provided on MATLAB Central
% https://uk.mathworks.com/matlabcentral/fileexchange/49908-adjusted-rand-index
% It has been slightly modified by Mike Croucher for speedup.

    %% Parse the input and throw errors
    % Check inputs
    adj = 0;
    if nargin == 0
        error('Arguments must be supplied.');
    end
    if nargin == 1
        error('Two partitions must be supplied.');
    end
    if nargin > 3
        error('Too many input arguments');
    end
    if nargin == 3
        if strcmp(varargin{1}, 'adjusted')
            adj = 1;
        else
            error('%s is an unrecognized argument.', varargin{1});
        end
    end
    if length(p1)~=length(p2)
        error('Both partitions must contain the same number of points.');
    end

    % Check if arguments need to be flattened
    if length(p1)~=numel(p1)
        p1 = p1(:);
        warning('The first partition was flattened to a 1D vector.')
    end
    if length(p2)~=numel(p2)
        p2 = p2(:);
        warning('The second partition was flattened to a 1D vector.')
    end

    % Check for integers
    if isreal(p1) && all(rem(p1, 1)==0)
        % all is well
    else
        warning('The first partition contains non-integers, which may make the results meaningless. Attempting to continue calculations.');
    end

    if isreal(p2) && all(rem(p2, 1)==0)
        % all is well
    else
        warning('The second partition contains non-integers, which may make the results meaningless. Attempting to continue calculations.');
    end

	%% Preliminary computations and cleansing of the partitions
    N = length(p1);
    [~, ~, p1] = unique(p1);
    N1 = max(p1);
    [~, ~, p2] = unique(p2);
    N2 = max(p2);

    n = zeros(N1,N2);
    %% Create the matching matrix
    % for i=1:1:N1
    %     for j=1:1:N2
    %         G1 = find(p1==i);
    %         G2 = find(p2==j);
    %         n(i,j) = length(intersect(G1,G2));
    %     end
    % end
    % Populate the contingency table
for i = 1:length(p1)
    n(p1(i), p2(i)) = n(p1(i), p2(i)) + 1;
end

    %% If required, calculate the basic rand index
    if adj==0
        ss = sum(sum(n.^2));
        ss1 = sum(sum(n,1).^2);
        ss2 =sum(sum(n,2).^2);
        ri = (nchoosek2(N,2) + ss - 0.5*ss1 - 0.5*ss2)/nchoosek2(N,2);
    end


    %% Otherwise, calculate the adjusted rand index
    if adj==1
        ssm = 0;
        sm1 = 0;
        sm2 = 0;
        for i=1:1:N1
            for j=1:1:N2
                ssm = ssm + nchoosek2(n(i,j),2);
            end
        end
        temp = sum(n,2);
        for i=1:1:N1
            sm1 = sm1 + nchoosek2(temp(i),2);
        end
        temp = sum(n,1);
        for i=1:1:N2
            sm2 = sm2 + nchoosek2(temp(i),2);
        end
        NN = ssm - sm1*sm2/nchoosek2(N,2);
        DD = (sm1 + sm2)/2 - sm1*sm2/nchoosek2(N,2);

        % Special case to handle perfect partitions
        if (NN == 0 && DD==0)
            ri = 1;
        else
            ri = NN/DD;
        end
    end


    %% Special definition of n choose k
    function c = nchoosek2(a,b)
        if a>1
            c = nchoosek(a,b);
        else
            c = 0;
        end
    end
end
