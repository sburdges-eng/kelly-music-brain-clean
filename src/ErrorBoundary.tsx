import { Component, ErrorInfo, ReactNode } from 'react';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
  }

  public render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: '20px', fontFamily: 'monospace' }}>
          <h1>Something went wrong</h1>
          <p>{this.state.error?.message}</p>
          <pre style={{ background: '#f0f0f0', padding: '10px', overflow: 'auto' }}>
            {this.state.error?.stack}
          </pre>
          <button onClick={() => window.location.reload()}>Reload App</button>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;

