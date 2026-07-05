import { lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Box, CircularProgress } from '@mui/material';
import { ColorModeProvider } from './hooks/useThemeMode';
import MainLayout from './layouts/MainLayout';
import ErrorBoundary from './components/Common/ErrorBoundary';

// Lazy load pages for code splitting & performance optimization
const Dashboard = lazy(() => import('./pages/Dashboard'));
const AmcPredictor = lazy(() => import('./pages/AmcPredictor'));
const ModelPerformance = lazy(() => import('./pages/ModelPerformance'));
const PipelineResults = lazy(() => import('./pages/PipelineResults'));
const About = lazy(() => import('./pages/About'));

// Centered loader fallback for Suspense
const PageLoader = () => (
  <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '50vh' }}>
    <CircularProgress size={40} />
  </Box>
);

function App() {
  return (
    <ColorModeProvider>
      <ErrorBoundary>
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<MainLayout />}>
              <Route
                index
                element={
                  <Suspense fallback={<PageLoader />}>
                    <Dashboard />
                  </Suspense>
                }
              />
              <Route
                path="predictor"
                element={
                  <Suspense fallback={<PageLoader />}>
                    <AmcPredictor />
                  </Suspense>
                }
              />
              <Route
                path="performance"
                element={
                  <Suspense fallback={<PageLoader />}>
                    <ModelPerformance />
                  </Suspense>
                }
              />
              <Route
                path="results"
                element={
                  <Suspense fallback={<PageLoader />}>
                    <PipelineResults />
                  </Suspense>
                }
              />
              <Route
                path="about"
                element={
                  <Suspense fallback={<PageLoader />}>
                    <About />
                  </Suspense>
                }
              />
              {/* Fallback redirect */}
              <Route
                path="*"
                element={
                  <Suspense fallback={<PageLoader />}>
                    <Dashboard />
                  </Suspense>
                }
              />
            </Route>
          </Routes>
        </BrowserRouter>
      </ErrorBoundary>
    </ColorModeProvider>
  );
}

export default App;
