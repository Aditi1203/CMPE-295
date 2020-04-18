function peaks = PeakDetection(x,ff,varargin)

N = length(x);
peaks = zeros(1,N);

th = .5;
rng = floor(th/ff);

if(nargin==3)
    disp(nargin)
    flag = varargin{1};
else
    flag = abs(max(x))>abs(min(x));
    disp(flag)
    disp(abs(max(x)))
    disp(abs(min(x)))
end

disp(flag)

if(flag)
    for j = 1 : N
        if(j>rng && j<N-rng)
            index = j-rng:j+rng;
            disp(index)
        elseif(j>rng)
            index = N-2*rng:N;
        else
            index = 1:2*rng;
        end

        if(max(x(index))==x(j))
            peaks(j) = 1;
        end
    end
else
    for j = 1 : N
        %         index = max(j-rng,1):min(j+rng,N);
        if(j>rng && j<N-rng)
            index = j-rng:j+rng;
        elseif(j>rng)
            index = N-2*rng:N;
        else
            index = 1:2*rng;
        end
        

        if(min(x(index))==x(j))
            peaks(j) = 1;
        end
    end
end


% % remove fake peaks
I = find(peaks);
d = diff(I);
 % z = find(d<rng);
peaks(I(d<rng))=0;
