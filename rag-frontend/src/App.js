import React, { useState } from 'react';

function App() {
	const [selectedFiles, setSelectedFiles] = useState([]);
	const [uploadMessage, setUploadMessage] = useState('');
	const [uploading, setUploading] = useState(false);

	const [query, setQuery] = useState('');
	const [answer, setAnswer] = useState('');
	const [querying, setQuerying] = useState(false);
	const [queryError, setQueryError] = useState('');

	const API_BASE_URL = 'http://127.0.0.1:8000'; // Make sure this matches your FastAPI server URL

	const handleFileChange = (event) => {
		setSelectedFiles(Array.from(event.target.files));
	};

	const handleUpload = async () => {
		if (selectedFiles.length === 0) {
			setUploadMessage('Please select at least one PDF file to upload.');
			return;
		}

		setUploading(true);
		setUploadMessage('Uploading and processing files...');
		const formData = new FormData();
		selectedFiles.forEach((file) => {
			formData.append('files', file);
		});

		try {
			const response = await fetch(`${API_BASE_URL}/upload`, {
				method: 'POST',
				body: formData,
			});

			const data = await response.json();

			if (response.ok) {
				setUploadMessage(
					`Upload successful! ${data.message} Total documents in store: ${data.total_documents_in_store}`
				);
				setSelectedFiles([]); // Clear selected files after successful upload
			} else {
				setUploadMessage(`Upload failed: ${data.detail || 'Unknown error'}`);
			}
		} catch (error) {
			console.error('Error uploading files:', error);
			setUploadMessage(`Error connecting to API: ${error.message}`);
		} finally {
			setUploading(false);
		}
	};

	const handleQuery = async () => {
		if (!query.trim()) {
			setQueryError('Please enter a question.');
			return;
		}

		setQuerying(true);
		setAnswer('Thinking...');
		setQueryError(''); // Clear previous errors

		try {
			const response = await fetch(`${API_BASE_URL}/query`, {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
				},
				body: JSON.stringify({ question: query }),
			});

			const data = await response.json();

			if (response.ok) {
				setAnswer(data.answer);
			} else {
				setAnswer(`Error: ${data.detail || 'Failed to get an answer.'}`);
				setQueryError(data.detail || 'An error occurred during query.');
			}
		} catch (error) {
			console.error('Error querying RAG:', error);
			setAnswer(`Error connecting to API: ${error.message}`);
			setQueryError(`Error connecting to API: ${error.message}`);
		} finally {
			setQuerying(false);
		}
	};

	return (
		<div style={styles.container}>
			<h1 style={styles.header}>Multimodal PDF RAG Chatbot</h1>

			<div style={styles.section}>
				<h2 style={styles.subHeader}>1. Upload PDF Documents</h2>
				<input
					type="file"
					accept=".pdf"
					multiple
					onChange={handleFileChange}
					style={styles.fileInput}
				/>
				<button onClick={handleUpload} disabled={uploading} style={styles.button}>
					{uploading ? 'Processing...' : 'Upload & Process PDFs'}
				</button>
				{selectedFiles.length > 0 && (
					<p style={styles.fileCount}>Selected: {selectedFiles.map(file => file.name).join(', ')}</p>
				)}
				{uploadMessage && <p style={styles.message}>{uploadMessage}</p>}
			</div>

			<div style={styles.section}>
				<h2 style={styles.subHeader}>2. Ask a Question</h2>
				<textarea
					value={query}
					onChange={(e) => setQuery(e.target.value)}
					placeholder="e.g., 'describe the hinge base' or 'what are the key findings?'"
					rows="4"
					style={styles.textarea}
				></textarea>
				<button onClick={handleQuery} disabled={querying} style={styles.button}>
					{querying ? 'Searching...' : 'Get Answer'}
				</button>
				{queryError && <p style={styles.errorMessage}>{queryError}</p>}
			</div>

			<div style={styles.section}>
				<h2 style={styles.subHeader}>Answer</h2>
				<div style={styles.answerBox}>
					{answer ? <p>{answer}</p> : <p>Your answer will appear here after querying.</p>}
				</div>
			</div>
		</div>
	);
}

const styles = {
	container: {
		fontFamily: 'Arial, sans-serif',
		maxWidth: '800px',
		margin: '30px auto',
		padding: '20px',
		border: '1px solid #ccc',
		borderRadius: '8px',
		boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
		backgroundColor: '#fff',
	},
	header: {
		textAlign: 'center',
		color: '#333',
		marginBottom: '30px',
	},
	section: {
		marginBottom: '25px',
		padding: '15px',
		border: '1px solid #eee',
		borderRadius: '6px',
		backgroundColor: '#f9f9f9',
	},
	subHeader: {
		color: '#555',
		borderBottom: '1px solid #ddd',
		paddingBottom: '10px',
		marginBottom: '15px',
	},
	fileInput: {
		display: 'block',
		marginBottom: '15px',
	},
	fileCount: {
		fontSize: '0.9em',
		color: '#666',
		marginTop: '10px',
		fontStyle: 'italic',
		wordBreak: 'break-all',
	},
	button: {
		backgroundColor: '#007bff',
		color: 'white',
		padding: '10px 20px',
		border: 'none',
		borderRadius: '5px',
		cursor: 'pointer',
		fontSize: '1em',
		transition: 'background-color 0.2s ease',
	},
	buttonHover: { // Note: This style is for conceptual hover. In React, you'd use CSS `:hover` or event handlers.
		backgroundColor: '#0056b3',
	},
	buttonDisabled: {
		backgroundColor: '#cccccc',
		cursor: 'not-allowed',
	},
	textarea: {
		width: '97%',
		padding: '10px',
		marginBottom: '15px',
		border: '1px solid #ddd',
		borderRadius: '4px',
		fontSize: '1em',
		minHeight: '80px',
		resize: 'vertical',
	},
	answerBox: {
		backgroundColor: '#e9ecef',
		padding: '15px',
		borderRadius: '6px',
		minHeight: '100px',
		border: '1px solid #dee2e6',
		whiteSpace: 'pre-wrap', // Preserves whitespace and line breaks
		wordWrap: 'break-word', // Breaks long words
	},
	message: {
		marginTop: '10px',
		fontSize: '0.9em',
		color: '#007bff',
	},
	errorMessage: {
		marginTop: '10px',
		fontSize: '0.9em',
		color: '#dc3545',
	}
};

export default App;
