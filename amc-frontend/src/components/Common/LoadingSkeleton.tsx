import React from 'react';
import { Box, Card, CardContent, Skeleton, Grid } from '@mui/material';

interface LoadingSkeletonProps {
  variant?: 'card' | 'table' | 'stats' | 'plot';
  count?: number;
  height?: number;
}

export const LoadingSkeleton: React.FC<LoadingSkeletonProps> = ({
  variant = 'card',
  count = 3,
  height,
}) => {
  if (variant === 'stats') {
    return (
      <Grid container spacing={3} sx={{ mb: 4, width: '100%' }}>
        {Array.from({ length: count }).map((_, i) => (
          <Grid size={{ xs: 12, sm: 6, md: 3 }} key={i}>
            <Card variant="outlined">
              <CardContent sx={{ textAlign: 'center', py: 2 }}>
                <Skeleton width="60%" sx={{ mx: 'auto', mb: 1 }} />
                <Skeleton variant="rectangular" height={40} width="40%" sx={{ mx: 'auto', borderRadius: 1 }} />
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    );
  }

  if (variant === 'table') {
    return (
      <Box sx={{ width: '100%', display: 'flex', flexDirection: 'column', gap: 1.5 }}>
        <Skeleton variant="rectangular" height={50} sx={{ borderRadius: 1 }} />
        {Array.from({ length: count }).map((_, i) => (
          <Skeleton key={i} variant="rectangular" height={40} sx={{ borderRadius: 1 }} />
        ))}
      </Box>
    );
  }

  if (variant === 'plot') {
    return (
      <Card variant="outlined" sx={{ width: '100%', height: height || 320, display: 'flex', flexDirection: 'column' }}>
        <CardContent sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', gap: 2, p: 2 }}>
          <Skeleton variant="text" width="40%" />
          <Skeleton variant="rectangular" height="80%" sx={{ borderRadius: 1 }} />
        </CardContent>
      </Card>
    );
  }

  // Default 'card' variant
  return (
    <Grid container spacing={3} sx={{ width: '100%' }}>
      {Array.from({ length: count }).map((_, i) => (
        <Grid size={{ xs: 12, md: 6, lg: 4 }} key={i}>
          <Card sx={{ height: '100%' }}>
            <CardContent sx={{ display: 'flex', flexDirection: 'column', gap: 2, p: 3 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Skeleton variant="circular" width={40} height={40} />
                <Skeleton variant="rectangular" width={60} height={24} sx={{ borderRadius: 4 }} />
              </Box>
              <Skeleton variant="text" width="80%" height={32} />
              <Skeleton variant="text" width="95%" />
              <Skeleton variant="text" width="60%" />
              <Skeleton variant="rectangular" height={1} sx={{ my: 1 }} />
              <Skeleton variant="text" width="40%" />
            </CardContent>
          </Card>
        </Grid>
      ))}
    </Grid>
  );
};

export default LoadingSkeleton;
