const InputForm = ({ variant, setVariant, onSubmit, disabled, errorMessage }) => {
  
    // Note: Variant validation (including mutalyzer lookup) is now handled by the backend
    const isFormValid = variant.refSeq && variant.gene && variant.codingChange;
  
    return (
      <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
        <h2 className="text-lg font-bold mb-4">Enter Variant Information</h2>
        {errorMessage && (
          <div className="mb-4 p-4 bg-red-100 border border-red-400 text-red-700 rounded">
            {errorMessage}
          </div>
        )}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Transcript ID *
            </label>
            <input
              type="text"
              placeholder="e.g., NM_000350.3"
              value={variant.refSeq}
              onChange={(e) => setVariant(prev => ({ ...prev, refSeq: e.target.value }))}
              disabled={disabled}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-100 disabled:cursor-not-allowed"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Gene Name *
            </label>
            <input
              type="text"
              placeholder="e.g., BRCA1"
              value={variant.gene}
              onChange={(e) => setVariant(prev => ({ ...prev, gene: e.target.value }))}
              disabled={disabled}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-100 disabled:cursor-not-allowed"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Coding Change *
            </label>
            <input
              type="text"
              placeholder="e.g., c.68_69del"
              value={variant.codingChange}
              onChange={(e) => setVariant(prev => ({ ...prev, codingChange: e.target.value }))}
              disabled={disabled}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-100 disabled:cursor-not-allowed"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Protein Change
            </label>
            <input
              type="text"
              placeholder="e.g., p.Glu23ValfsTer17"
              value={variant.proteinChange}
              onChange={(e) => setVariant(prev => ({ ...prev, proteinChange: e.target.value }))}
              disabled={disabled}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-100 disabled:cursor-not-allowed"
            />
          </div>
        </div>
        <div className="flex justify-end mt-4">
          <button
            onClick={onSubmit}
            disabled={disabled || !isFormValid}
            className="px-4 py-2 bg-black text-white rounded-md hover:bg-gray-700 flex items-center"
          >
            Analyze Variant
          </button>
        </div>
      </div>
    );
};

export default InputForm;