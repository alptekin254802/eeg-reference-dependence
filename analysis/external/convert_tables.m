function convert_tables(manifest_path)
% Convert native MATLAB tables and retain acquisition metadata.
rows = jsondecode(fileread(manifest_path));
for j = 1:numel(rows)
    source = load(rows(j).input_path);
    names = fieldnames(source); found = {};
    for k = 1:numel(names)
        if istable(source.(names{k})), found{end+1} = names{k}; end
    end
    assert(numel(found)==1,'Expected one table per recording.');
    t=source.(found{1}); assert(width(t)==21,'Expected 21 columns.');
    eeg=double(t{:,1:19}); time_seconds=double(t{:,20});
    channel_names=t.Properties.VariableNames(1:19);
    metadata=struct('variable_names',{t.Properties.VariableNames}, ...
        'variable_units',{t.Properties.VariableUnits}, ...
        'variable_descriptions',{t.Properties.VariableDescriptions}, ...
        'description',t.Properties.Description,'userdata',t.Properties.UserData);
    save(rows(j).output_path,'eeg','time_seconds','channel_names','metadata','-v7');
end
end
