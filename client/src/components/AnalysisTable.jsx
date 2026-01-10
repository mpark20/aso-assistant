import React, {useState} from 'react';
import ReactMarkdown from 'react-markdown';
import { Save, X, Edit2, RefreshCw, FileText, Code } from 'lucide-react';

// Modal for displaying the full report when expanded
const ReportModal = ({ report, onClose }) => {
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-3xl w-full max-h-[80vh] flex flex-col">
        <div className="flex items-center justify-between p-4 border-b">
          <h2 className="text-xl font-semibold">Full Report</h2>
          <button
            onClick={onClose}
            className="p-1 hover:bg-gray-100 rounded-full transition-colors"
            aria-label="Close"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
        <div className="p-6 overflow-y-auto flex-1 whitespace-pre-wrap">
          <div className="">
            <ReactMarkdown>{report || 'No report available.'}</ReactMarkdown>
          </div>
        </div>
      </div>
    </div>
  );
};

// Modal for displaying the full payload dump
const PayloadModal = ({ payload, onClose }) => {
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-5xl w-full max-h-[90vh] flex flex-col">
        <div className="flex items-center justify-between p-4 border-b">
          <h2 className="text-xl font-semibold">Full Session History</h2>
          <button
            onClick={onClose}
            className="p-1 hover:bg-gray-100 rounded-full transition-colors"
            aria-label="Close"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
        <div className="p-6 overflow-y-auto flex-1">
          <pre className="bg-gray-50 p-4 rounded-lg text-sm overflow-x-auto whitespace-pre-wrap">
            {payload ? JSON.stringify(payload, null, 2) : 'No payload available.'}
          </pre>
        </div>
      </div>
    </div>
  );
};

const AnalysisTableRow = ({ subtask, stepId, cellData, isEditing, editValue, onEditChange, onEdit, onSave, onCancel, onShowModal, onShowPayload, onRetry, isLoading,  }) => {
  return (
    <tr className="hover:bg-gray-50">
      <td className="border p-3 font-medium">{subtask.name}</td>
      <td className="border p-3 max-w-md">
        {isEditing ? (
          <textarea
            value={editValue}
            onChange={(e) => onEditChange(e.target.value)}
            className="w-full p-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
            rows={3}
          />
        ) : (
          <div className="space-y-2">
            <div className="text-sm">
              {isLoading ? (
                <div className="flex items-center space-x-2">
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>{cellData.status} ...</span>
                </div>
              ) : (
                cellData.value || `${cellData.status} ...`
              )}
            </div>
            {!isLoading && cellData.value && cellData.report && (
              <button
                onClick={onShowModal}
                className="flex items-center space-x-1 text-sm text-blue-600 hover:text-blue-800 hover:underline"
              >
                <FileText className="w-4 h-4" />
                <span>Show full report</span>
              </button>
            )}
            {!isLoading && cellData.fullPayload && (
              <button
                onClick={onShowPayload}
                className="flex items-center space-x-1 text-sm text-purple-600 hover:text-purple-800 hover:underline mt-2"
              >
                <Code className="w-4 h-4" />
                <span>View payload dump</span>
              </button>
            )}
          </div>
        )}
      </td>
      <td className="border p-3 text-sm text-blue-600 max-w-xs truncate" title={cellData.sources}>
        <ReactMarkdown>{cellData.sources || 'N/A'}</ReactMarkdown>
      </td>
      <td className="border p-3">
        <div className="flex space-x-2">
          {isEditing ? (
            <>
              <button onClick={onSave} className="p-1 text-green-600 hover:bg-green-100 rounded">
                <Save className="w-4 h-4" />
              </button>
              <button onClick={onCancel} className="p-1 text-red-600 hover:bg-red-100 rounded">
                <X className="w-4 h-4" />
              </button>
            </>
          ) : (
            <>
            <button onClick={() => onEdit(stepId, subtask.name)} className="p-1 text-gray-600 hover:bg-gray-100 rounded">
              <Edit2 className="w-4 h-4" />
            </button>
            <button onClick={() => onRetry(stepId, subtask.name)} className="p-1 text-gray-600 hover:bg-gray-100 rounded">
              <RefreshCw className="w-4 h-4" />
            </button>
            </>
          )}
        </div>
      </td>
    </tr>
  );
};

// ============ Analysis Table Component ============
const AnalysisTable = ({
  step,
  data,
  editingCell,
  editValue,
  onEditChange,
  onEdit,
  onSave,
  onCancel,
  onRetry,
  loadingCells
}) => {
  const [modalReport, setModalReport] = useState(null);
  const [modalPayload, setModalPayload] = useState(null);

  return (
    <>
      <div className="overflow-x-auto">
        <table className="w-full border-collapse">
          <thead>
            <tr className="bg-gray-50">
              <th className="border p-3 text-left font-medium">Analysis Component</th>
              <th className="border p-3 text-left font-medium">Result</th>
              {/* <th className="border p-3 text-left font-medium">Confidence</th> */}
              <th className="border p-3 text-left font-medium">Sources</th>
              <th className="border p-3 text-left font-medium">Actions</th>
            </tr>
          </thead>
          <tbody>
            {step.subtasks.map((subtask, subIndex) => {
              const cellData = data[step.id]?.[subtask.name] || {};
              const isEditing = editingCell?.stepId === step.id && editingCell?.field === subtask.name;
              const isLoading = loadingCells?.has(`${step.id}-${subtask.name}`);
              
              return (
                <AnalysisTableRow
                  key={subIndex}
                  subtask={subtask}
                  stepId={step.id}
                  cellData={cellData}
                  isEditing={isEditing}
                  editValue={editValue}
                  onEditChange={onEditChange}
                  onEdit={onEdit}
                  onSave={onSave}
                  onCancel={onCancel}
                  onRetry={onRetry}
                  onShowModal={() => setModalReport(cellData.report)}
                  onShowPayload={() => setModalPayload(cellData.fullPayload)}
                  isLoading={isLoading}
                />
              );
            })}
          </tbody>
        </table>
      </div>
      
      {modalReport && (
        <ReportModal 
          report={modalReport} 
          onClose={() => setModalReport(null)} 
        />
      )}
      
      {modalPayload && (
        <PayloadModal 
          payload={modalPayload} 
          onClose={() => setModalPayload(null)} 
        />
      )}
    </>
  );
};

export default AnalysisTable;