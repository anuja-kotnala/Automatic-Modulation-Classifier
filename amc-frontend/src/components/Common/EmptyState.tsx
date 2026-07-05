import React from 'react';
import { Box, Typography, Paper } from '@mui/material';
import { InboxOutlined as DefaultEmptyIcon } from '@mui/icons-material';

interface EmptyStateProps {
  title: string;
  description: string;
  icon?: React.ReactNode;
  action?: React.ReactNode;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  title,
  description,
  icon,
  action,
}) => {
  return (
    <Box
      sx={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        width: '100%',
        py: 6,
        px: 2,
      }}
    >
      <Paper
        variant="outlined"
        sx={{
          p: 4,
          width: '100%',
          maxWidth: 500,
          textAlign: 'center',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 2,
          borderColor: 'divider',
        }}
      >
        <Box sx={{ color: 'text.secondary', opacity: 0.6 }}>
          {icon || <DefaultEmptyIcon sx={{ fontSize: 60 }} />}
        </Box>
        <Typography variant="h6" sx={{ fontWeight: 700, color: 'text.primary' }}>
          {title}
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ maxWidth: 360, mx: 'auto' }}>
          {description}
        </Typography>
        {action && <Box sx={{ mt: 1 }}>{action}</Box>}
      </Paper>
    </Box>
  );
};

export default EmptyState;
