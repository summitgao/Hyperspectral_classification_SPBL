function out = checkfield(obj,obj_name,field,lb,ub)
    if ~isfield(obj,field)
        error('Checkfield error: %s.%s must be specified!',obj_name,field);
    end
    if nargin >= 4
        if ~isempty(find(obj.(field)<lb,1))
            error('Checkfield error: %s.%s must all > %.2f.',obj_name,field,lb);
        end
    end
    if nargin >= 5
        if ~isempty(find(obj.(field)>ub,1))
            error('Checkfield error: %s.%s must all < %.2f.',obj_name,field,ub);
        end
    end
    out = 1;
end