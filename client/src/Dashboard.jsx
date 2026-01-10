import React, { useState, useEffect } from 'react';
import AnalysisTable from './components/AnalysisTable';
import InputForm from './components/InputForm';
import { ChevronRight, ChevronDown, Check, AlertCircle, Database, FileText, Microscope, Beaker, RefreshCw, CheckCircle } from 'lucide-react';

const API_BASE_URL = 'http://localhost:8000';

// Icon mapping
const ICON_MAP = {
  'Database': Database,
  'FileText': FileText,
  'Microscope': Microscope,
  'Beaker': Beaker
};

// Utility Components
const StatusIcon = ({ status }) => {
  switch (status) {
    case 'completed':
      return <Check className="w-4 h-4 text-green-600" />;
    case 'in-progress':
      return <RefreshCw className="w-4 h-4 text-blue-600" />;
    case 'pending':
      return <AlertCircle className="w-4 h-4 text-gray-400" />;
    case 'ready':
      return <CheckCircle className="w-4 h-4 text-yellow-600" />;
    default:
      return 'not started';
  }
};

const ProgressBar = ({ progress, className = "" }) => (
  <div className={`bg-gray-200 rounded-full h-2 ${className}`}>
    <div 
      className="h-2 rounded-full transition-all duration-300 bg-black"
      style={{ width: `${progress}%` }}
    />
  </div>
);

const PaperCard = ({ paper }) => {
  const tagColors = {
    "closed access": "bg-red-100 text-red-800",
    "abstract": "bg-blue-100 text-blue-800",
    "full text": "bg-green-100 text-green-800",
    "default": "bg-gray-100 text-gray-800",
  }
  let tag = "abstract only";
  if (paper.content.length > 0) {
    tag = "full text";
  }
  return (
    <div className="bg-white rounded-lg shadow-sm p-4">
      <h3 className="text-lg font-medium">{paper.title}</h3>
      <p className="text-sm text-gray-500">{paper.journal}</p>
      <p className="text-sm text-gray-500">{paper.date}</p>
      <a className="text-sm text-blue-500" href={paper.link} target="_blank">{paper.link}</a>
      <p className="text-sm text-gray-500">{paper.snippet.slice(0, 50)} {paper.snippet.length > 50 ? "..." : ""}</p>
      <p className="text-sm text-gray-500">Found in {paper.source}</p>
      <span className={`inline-block px-2 py-1 ${tagColors[tag] || tagColors["default"]} rounded-full text-xs font-medium`}>
        {tag}
      </span>
    </div>
  );
};

// Workflow Step
const WorkflowStep = ({ 
  step,
  isExpanded,
  onToggle,
  data, 
  editingCell,
  editValue,
  onEditChange,
  onEdit,
  onSave,
  onCancel,
  onRetry,
  loadingCells,
  onAccept,
  showAccept
}) => {
  const IconComponent = ICON_MAP[step.iconName] || Database;
  
  return (
    <div className="bg-white rounded-lg shadow-sm">
      <div className="p-4 border-b cursor-pointer hover:bg-gray-50" onClick={onToggle}>
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            {isExpanded ? <ChevronDown className="w-5 h-5" /> : <ChevronRight className="w-5 h-5" />}
            {/* <IconComponent className="w-4 h-4" /> */}
            <h3 className="text-lg font-medium">{step.title}</h3>
            <StatusIcon status={step.status} />
          </div>
          <ProgressBar progress={step.progress} className="w-32" />
        </div>
      </div>
      {isExpanded && (
        <div className="p-6">
          <AnalysisTable
            step={step}
            data={data}
            editingCell={editingCell}
            editValue={editValue}
            onEditChange={onEditChange}
            onEdit={onEdit}
            onSave={onSave}
            onCancel={onCancel}
            onRetry={onRetry}
            loadingCells={loadingCells}
          />
          
          {showAccept && step.status === 'ready' && (
            <div className="mt-4 flex justify-end">
              <button
                onClick={onAccept}
                className="px-6 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 flex items-center space-x-2"
              >
                <CheckCircle className="w-5 h-5" />
                <span>Accept & Continue to Next Step</span>
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// Main Dashboard
const Dashboard = () => {
  const [variant, setVariant] = useState({ refSeq: 'NM_000350.3', gene: 'ABCA4', codingChange: 'c.2626C>T', proteinChange: '' });
  const [expandedSteps, setExpandedSteps] = useState(new Set());
  const [editingCell, setEditingCell] = useState(null);
  const [editValue, setEditValue] = useState('');
  const [loadingCells, setLoadingCells] = useState(new Set());
  const [errorMessage, setErrorMessage] = useState('');
  const [steps, setSteps] = useState([]);
  const [data, setData] = useState({});
  const [configLoaded, setConfigLoaded] = useState(false);
  const [currentStepIndex, setCurrentStepIndex] = useState(-1);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [papers, setPapers] = useState([]);
  const [papersLoading, setPapersLoading] = useState(false);
  const [showAllPapers, setShowAllPapers] = useState(false);

  useEffect(() => {
    loadConfig();
  }, []);


  const getIconNameForStep = (title) => {
    if (title.includes('Disease')) return 'Database';
    if (title.includes('Splicing')) return 'Microscope';
    if (title.includes('Exon')) return 'FileText';
    if (title.includes('Therapeutic')) return 'Beaker';
    return 'Database';
  };

  /**
   * Handles the start button. Validates input and triggers the first step.
   */
  const handleStart = async () => {
    setVariant({
      refSeq: variant.refSeq.trim(),
      gene: variant.gene.trim(),
      codingChange: variant.codingChange.trim(),
      proteinChange: variant.proteinChange.trim()
    });
    if (!variant.gene || !variant.codingChange || !variant.refSeq) {
      setErrorMessage('Please fill in all required fields');
      return;
    }
    setErrorMessage('');
    // Note: Validation is now handled by the backend via mutalyzer
    // Papers fetching removed - can be added back if needed
    setCurrentStepIndex(0);
    await runStep(0);
  };

  const handleRetry = async (stepId, subtaskName) => {
    // TODO: handle adding additional feedback/resources before the rerun.
    setData(prev => {
      const updatedStep = { ...prev[stepId] };
      delete updatedStep[subtaskName];
      const newData = {
        ...prev,
        [stepId]: updatedStep,
      };
      runSubtaskStream(stepId, subtaskName, newData);
      return newData;
    });
  };

  /**
   * Handles the edit button for a step. This sets the cell for the given step and subtask to be editable.
   * @param {number} stepId 
   * @param {string} field 
   */
  const handleEdit = (stepId, field) => {
    setEditingCell({ stepId, field });
    setEditValue(data[stepId]?.[field]?.value || '');
  };

  /**
   * Handles the save button for a step. This saves the user's input for a cell to the app state.
   */
  const handleSave = () => {
    const { stepId, field } = editingCell;
    setData(prev => ({
      ...prev,
      [stepId]: {
        ...prev[stepId],
        [field]: {
          ...prev[stepId]?.[field],
          value: editValue,
          lastUpdated: new Date().toISOString().split('T')[0]
        }
      }
    }));
    setEditingCell(null);
    setEditValue('');
  };

  /**
   * Handles the cancel button for a step.
   */
  const handleCancel = () => {
    setEditingCell(null);
    setEditValue('');
  };

  /**
   * Handles the accept button for a step.
   * Marks the step as completed and runs the next step if it exists.
   */
  const handleAcceptStep = async () => {
    const nextStepIndex = currentStepIndex + 1;
    
    setSteps(prev => prev.map((s, idx) => 
      idx === currentStepIndex ? { ...s, status: 'completed' } : s
    ));
    
    if (nextStepIndex < steps.length) {
      setCurrentStepIndex(nextStepIndex);
      await runStep(nextStepIndex);
    } else {
      console.log('All steps complete');
    }
  };

  // Services for communication with the server

  /**
   * Loads the configuration for each step in the workflow.
   * Steps are now hardcoded based on the backend protocol.
   */
  const loadConfig = async () => {
    try {
      // Define steps based on the backend protocol
      const stepsConfig = [
        {
          id: 0,
          title: "Inheritance Pattern",
          subtasks: [{ name: "inheritance_pattern" }]
        },
        {
          id: 1,
          title: "Pathomechanism",
          subtasks: [{ name: "pathomechanism" }]
        },
        {
          id: 2,
          title: "Dosage Sensitivity",
          subtasks: [{ name: "dosage_sensitivity" }]
        },
        {
          id: 3,
          title: "Splicing Effects - Identification",
          subtasks: [{ name: "splicing_effects::identification" }]
        },
        {
          id: 4,
          title: "Splicing Effects - Categorization",
          subtasks: [{ name: "splicing_effects::categorization" }]
        },
        {
          id: 5,
          title: "ASO Check",
          subtasks: [{ name: "aso_check" }]
        }
      ];
      
      // add status and progress to each step, as this is stored client side.
      const stepsWithState = stepsConfig.map(step => ({
        ...step,
        status: 'pending',
        progress: 0,
        subtasks: step.subtasks.map(subtask => ({
          ...subtask,
          status: 'pending',
          progress: 0
        })),
        iconName: getIconNameForStep(step.title)
      }));
      
      setSteps(stepsWithState);
      setConfigLoaded(true);
    } catch (error) {
      setErrorMessage(`Failed to load configuration: ${error.message}`);
    }
  };


  const runSubtaskStream = async (stepIndex, subtaskName) => {
    const cellKey = `${stepIndex}-${subtaskName}`;
    
    setLoadingCells(prev => new Set([...prev, cellKey]));
    // update the status of the current subtask
    setData(prev => ({
      ...prev,
      [stepIndex]: {
        ...prev[stepIndex],
        [subtaskName]: { ...prev[stepIndex]?.[subtaskName], status: 'Starting analysis' }
      }
    }));
    
    try {
      // Map variant fields to backend format
      const variantRequest = {
        transcript: variant.refSeq,
        coding_change: variant.codingChange,
        gene: variant.gene
      };

      const response = await fetch(`${API_BASE_URL}/chat?stream=true`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          step: subtaskName,
          variant: variantRequest
        })
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Analysis failed');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let finalAnswer = '';
      const allPayloads = []; // Store all payloads for this subtask

      // Process NDJSON stream
      while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.trim()) continue;
          
          try {
            const payload = JSON.parse(line);
            allPayloads.push(payload); // Store every payload
            
            switch (payload.type) {
              case 'generation':
                // Update status with generation progress
                setData(prev => ({
                  ...prev,
                  [stepIndex]: {
                    ...prev[stepIndex],
                    [subtaskName]: { 
                      ...prev[stepIndex]?.[subtaskName], 
                      status: `Thinking... `,
                      fullPayload: allPayloads
                    }
                  }
                }));
                break;
              
              case 'tool_output':
                // Update status after tool execution
                setData(prev => ({
                  ...prev,
                  [stepIndex]: {
                    ...prev[stepIndex],
                    [subtaskName]: { 
                      ...prev[stepIndex]?.[subtaskName], 
                      status: `Running ${payload.tool_name}`,
                      fullPayload: allPayloads
                    }
                  }
                }));
                break;
                
              case 'final':
                // Final result - update data state
                finalAnswer = payload.final_answer || payload.generated_text || '';
                setData(prev => ({
                  ...prev,
                  [stepIndex]: {
                    ...prev[stepIndex],
                    [subtaskName]: {
                      value: finalAnswer.slice(0, 500) + '...',
                      report: payload.generated_text || '',
                      sources: payload.sources || '',
                      status: 'completed',
                      lastUpdated: new Date().toISOString().split('T')[0],
                      fullPayload: allPayloads
                    }
                  }
                }));
                break;
                
              case 'error':
                setData(prev => ({
                  ...prev,
                  [stepIndex]: {
                    ...prev[stepIndex],
                    [subtaskName]: {
                      ...prev[stepIndex]?.[subtaskName],
                      status: 'error',
                      value: '',
                      report: '',
                      fullPayload: allPayloads
                    }
                  }
                }));
                throw new Error(payload.error || 'Analysis failed');
                
              case 'stopped':
                // Handle stopped state
                if (finalAnswer) {
                  setData(prev => ({
                    ...prev,
                    [stepIndex]: {
                      ...prev[stepIndex],
                      [subtaskName]: {
                        value: finalAnswer,
                        report: finalAnswer,
                        sources: payload.tool_calls ? JSON.stringify(payload.tool_calls, null, 2) : '',
                        status: 'completed',
                        lastUpdated: new Date().toISOString().split('T')[0],
                        fullPayload: allPayloads
                      }
                    }
                  }));
                }
                break;
            }
          } catch (parseError) {
            // Skip malformed JSON lines
            console.warn('Failed to parse JSON line:', line, parseError);
          }
        }
      }
      
      // Ensure final payloads are stored even if stream ends without 'final' type
      if (allPayloads.length > 0) {
        setData(prev => ({
          ...prev,
          [stepIndex]: {
            ...prev[stepIndex],
            [subtaskName]: {
              ...prev[stepIndex]?.[subtaskName],
              fullPayload: allPayloads
            }
          }
        }));
      }
      
      setErrorMessage('');
    } catch (error) {
      setErrorMessage(`Failed to analyze ${subtaskName}: ${error.message}`);
      setData(prev => ({
        ...prev,
        [stepIndex]: {
          ...prev[stepIndex],
          [subtaskName]: { 
            ...prev[stepIndex]?.[subtaskName],
            status: `Error: ${error.message}`,
            value: '', report: '', sources: '', lastUpdated: '',
            fullPayload: prev[stepIndex]?.[subtaskName]?.fullPayload || []
          }
        }
      }));
    } finally {
      setLoadingCells(prev => {
        const newSet = new Set(prev);
        newSet.delete(cellKey);
        return newSet;
      });
    }
  };

  /**
   * Runs a step of the workflow. Sends user input and current step/subtask state to the server,
   * and progressively updates the app state after each subtask.
   * @param {number} stepIndex 
   */
  const runStep = async (stepIndex) => {
    const step = steps[stepIndex];
    setIsAnalyzing(true);
    setExpandedSteps(prev => new Set([...prev, stepIndex]));
    
    setSteps(prev => prev.map((s, idx) => 
      idx === stepIndex ? { ...s, status: 'in-progress' } : s
    ));

    // send each subtask to the server, await results, and update the app state.
    // await Promise.all(step.subtasks.map(subtask => runSubtask(step.id, subtask.name, data)));
    await Promise.all(step.subtasks.map(subtask => runSubtaskStream(step.id, subtask.name)));
    
    setSteps(prev => prev.map((s, idx) => 
      idx === stepIndex ? { ...s, status: 'ready', progress: 100 } : s
    ));
    
    setIsAnalyzing(false);
  };

  if (!configLoaded) {
    return (
      <div className="min-h-screen bg-gray-50 py-8 px-4 sm:px-6 lg:px-8">
        <div className="flex items-center space-x-2">
          <RefreshCw className="w-6 h-6 animate-spin" />
          <span>Loading configuration...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4 sm:px-6 lg:px-8">
      <h1 className="text-3xl font-bold mb-6">Variant Analysis Dashboard</h1>
      <InputForm 
        variant={variant}
        setVariant={setVariant}
        onSubmit={handleStart}
        disabled={isAnalyzing || papersLoading || currentStepIndex >= 0}
        errorMessage={errorMessage}
      />

      {papersLoading && (
        <div className="mt-8 mb-8">
          <h2 className="text-xl mb-2">Loading...</h2>
        </div>
      )}

      {/* {papers.length > 0 && (
        <div className="mt-8 mb-8">
          <h2 className="text-xl font-bold mb-2">Papers</h2>
          <div className="space-y-2">
            {papers.map((paper, idx) => (
              <PaperCard key={idx} paper={paper} />
            ))}
          </div>
        </div>
      )} */}
      {papers.length > 0 && (
        <div className="mt-8 mb-8">
          <h2 className="text-xl font-bold mb-2">Papers</h2>
          <div className="space-y-2">
            {(showAllPapers ? papers : papers.slice(0, 10)).map((paper, idx) => (
              <PaperCard key={idx} paper={paper} />
            ))}
          </div>
          {papers.length > 10 && (
            <div className="mt-4">
              <button
                onClick={() => setShowAllPapers(!showAllPapers)}
                className="px-4 py-2 text-sm bg-gray-200 rounded hover:bg-gray-300"
              >
                {showAllPapers ? "Show Less" : "Show More"}
              </button>
            </div>
          )}
        </div>
      )}
        
      {currentStepIndex >= 0 && (
        <div className="space-y-4">
        {steps.map((step, idx) => (
          <WorkflowStep
            key={step.id}
            step={step}
            isExpanded={expandedSteps.has(idx)}
            onToggle={() => setExpandedSteps(prev => prev.has(idx) ? new Set([...prev].filter(i => i !== idx)) : new Set([...prev, idx]))}
            data={data}
            editingCell={editingCell}
            editValue={editValue} 
            onEditChange={setEditValue}
            onEdit={handleEdit}
            onSave={handleSave}
            onCancel={handleCancel}
            onRetry={handleRetry}
            loadingCells={loadingCells}
            onAccept={handleAcceptStep}
            showAccept={idx === currentStepIndex}
          />
        ))}
      </div>
      )}
    </div>
  );
};

export default Dashboard;