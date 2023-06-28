import React, { useState } from 'react';

const layerTypes = ['Dense', 'Output', 'Activation', 'Pooling', 'Filter', 'Dropout'];
const activationTypes = ['ReLU', 'Softmax'];

const CNNCustomization = () => {
    const [numLayers, setNumLayers] = useState(1);
    const [layers, setLayers] = useState([
        {
            id: 1,
            type: 'Dense',
            parameters: { nodes: 32 },
        },
    ]);

    const handleAddLayer = () => {
        setNumLayers(numLayers + 1);
        setLayers([...layers, { id: numLayers + 1, type: 'Dense', parameters: {} }]);
    };

    const handleLayerTypeChange = (event, layerId) => {
        const updatedLayers = layers.map((layer) =>
            layer.id === layerId ? { ...layer, type: event.target.value } : layer
        );
        setLayers(updatedLayers);
    };

    const handleParameterChange = (event, layerId, parameterName) => {
        const updatedLayers = layers.map((layer) =>
            layer.id === layerId
                ? {
                    ...layer,
                    parameters: {
                        ...layer.parameters,
                        [parameterName]: event.target.value,
                    },
                }
                : layer
        );
        setLayers(updatedLayers);
    };

    return (
        <div>
            <h1>Customize your CNN</h1>
            <p>Number of Layers: {numLayers}</p>
            <button onClick={handleAddLayer}>Add Layer</button>
            {layers.map((layer) => (
                <div key={layer.id}>
                    <label htmlFor={`layer-type-${layer.id}`}>Layer {layer.id} Type:</label>
                    <select
                        id={`layer-type-${layer.id}`}
                        value={layer.type}
                        onChange={(event) => handleLayerTypeChange(event, layer.id)}
                    >
                        {layerTypes.map((type) => (
                            <option key={type} value={type}>
                                {type}
                            </option>
                        ))}
                    </select>
                    {layer.type === 'Dense' && (
                        <>
                            <label htmlFor={`layer-${layer.id}-nodes`}>Nodes:</label>
                            <input
                                id={`layer-${layer.id}-nodes`}
                                type="number"
                                value={layer.parameters.nodes}
                                onChange={(event) =>
                                    handleParameterChange(event, layer.id, 'nodes')
                                }
                            />
                        </>
                    )}
                    {layer.type === 'Filter' && (
                        <>
                            <label htmlFor={`layer-${layer.id}-channels`}>Channels:</label>
                            <input
                                id={`layer-${layer.id}-channels`}
                                type="number"
                                value={layer.parameters.channels}
                                onChange={(event) =>
                                    handleParameterChange(event, layer.id, 'channels')
                                }
                            />
                            <label htmlFor={`layer-${layer.id}-dimensions`}>Dimensions:</label>
                            <input
                                id={`layer-${layer.id}-dimensions`}
                                type="number"
                                value={layer.parameters.dimensions}
                                onChange={(event) =>
                                    handleParameterChange(event, layer.id, 'dimensions')
                                }
                            />
                        </>
                    )}
                    {layer.type === 'Activation' && (
                        <>
                            <label htmlFor={`layer-${layer.id}-activation`}>Activation:</label>
                            <select
                                id={`layer-${layer.id}-activation`}
                                value={layer.parameters.activation}
                                onChange={(event) =>
                                    handleParameterChange(event, layer.id, 'activation')
                                }
                            >
                                {activationTypes.map((type) => (
                                    <option key={type} value={type}>
                                        {type}
                                    </option>
                                ))}
                            </select>
                        </>
                    )}
                    {layer.type === 'Output' && (
                        <>
                            <label htmlFor={`layer-${layer.id}-nodes`}>Nodes:</label>
                            <input
                                id={`layer-${layer.id}-nodes`}
                                type="number"
                                value={layer.parameters.nodes}
                                onChange={(event) =>
                                    handleParameterChange(event, layer.id, 'nodes')
                                }
                            />
                        </>
                    )}
                    {layer.type === 'Dropout' && (
                        <>
                            <label htmlFor={`layer-${layer.id}-dropout`}>Dropout Rate:</label>
                            <input
                                id={`layer-${layer.id}-dropout`}
                                type="number"
                                value={layer.parameters.dropout}
                                onChange={(event) =>
                                    handleParameterChange(event, layer.id, 'dropout')
                                }
                            />
                        </>
                    )}
                </div>
            ))}
        </div>
    );
};

export default CNNCustomization;
