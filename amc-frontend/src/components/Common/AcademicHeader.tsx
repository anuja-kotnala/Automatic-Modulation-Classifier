import React from 'react';
import { Box, Typography, Divider } from '@mui/material';

interface AcademicHeaderProps {
  title: string;
  subtitle: string;
  action?: React.ReactNode;
}

export const AcademicHeader: React.FC<AcademicHeaderProps> = ({ title, subtitle, action }) => {
  return (
    <Box sx={{ mb: 4 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 2 }}>
        <Box sx={{ flexGrow: 1 }}>
          <Typography variant="h4" sx={{ fontWeight: 800, color: 'text.primary', letterSpacing: -0.5 }}>
            {title}
          </Typography>
          <Typography variant="body1" color="text.secondary" sx={{ mt: 0.5, maxWidth: 800 }}>
            {subtitle}
          </Typography>
        </Box>
        {action && <Box sx={{ display: 'flex', alignItems: 'center' }}>{action}</Box>}
      </Box>
      <Divider sx={{ mt: 3, mb: 1 }} />
    </Box>
  );
};

export default AcademicHeader;
