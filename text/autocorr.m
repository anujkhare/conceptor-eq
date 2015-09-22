function autocorrPlotData = autocorr(ts, maxLag)
    % on input of a scalar timeseries ts, create autocorrelation for lags 0, 1,
    % ... , maxLag
    % input ts may be in column or row format
    if length(ts) <= maxLag
        error('timeseries too short for computing requested autocorrs');
    end

    % make ts column ts if not already in that format

    if size(ts, 1) == 1
        tsCol = ts';
    else
        tsCol = ts;
    end

    L = size(tsCol,1);
    dataLagMat = zeros(L-maxLag, maxLag+1);
    dataLagMat(:,1) = tsCol(1:L-maxLag,1);
    for lag = 1:maxLag
        dataLagMat(:,lag+1) = tsCol(1+lag:L-maxLag+lag, 1);
    end
    dataMat = repmat(tsCol(1:L-maxLag,1),1, maxLag+1);

    autocorrPlotData = diag(dataLagMat' * dataMat) / (L-maxLag);
end
