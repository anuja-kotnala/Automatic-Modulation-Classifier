import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation, Outlet } from 'react-router-dom';
import {
  AppBar,
  Box,
  CssBaseline,
  Divider,
  Drawer,
  IconButton,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Toolbar,
  Typography,
  Chip,
  Tooltip,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Dashboard as DashboardIcon,
  SettingsInputComponent as SignalIcon,
  BarChart as ChartIcon,
  Brightness4 as DarkIcon,
  Brightness7 as LightIcon,
  Wifi as OnlineIcon,
  WifiOff as OfflineIcon,
  FolderOpen as FolderIcon,
  Info as InfoIcon,
} from '@mui/icons-material';
import { useColorMode } from '../hooks/useThemeMode';
import { amcApi } from '../services/amcApi';

const drawerWidth = 260;

export const MainLayout: React.FC = () => {
  const { mode, toggleColorMode } = useColorMode();
  const navigate = useNavigate();
  const location = useLocation();

  const [mobileOpen, setMobileOpen] = useState(false);
  const [mlOnline, setMlOnline] = useState<boolean | null>(null);
  const [dlOnline, setDlOnline] = useState<boolean | null>(null);

  const checkHealth = async () => {
    try {
      const res = await amcApi.getHealth();
      setMlOnline(res.status === 'ok');
    } catch (e) {
      setMlOnline(false);
    }
    try {
      const res = await amcApi.getHealthDL();
      setDlOnline(res.status === 'ok');
    } catch (e) {
      setDlOnline(false);
    }
  };

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 15000); // Check health every 15s
    return () => clearInterval(interval);
  }, []);

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const navItems = [
    { text: 'Dashboard', path: '/', icon: <DashboardIcon /> },
    { text: 'AMC Predictor', path: '/predictor', icon: <SignalIcon /> },
    { text: 'Model Performance', path: '/performance', icon: <ChartIcon /> },
    { text: 'Pipeline Results', path: '/results', icon: <FolderIcon /> },
    { text: 'About', path: '/about', icon: <InfoIcon /> },
  ];

  const drawer = (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Toolbar sx={{ justifyContent: 'center', py: 3.5 }}>
        <Typography
          variant="h5"
          component="div"
          sx={{
            fontWeight: 800,
            letterSpacing: '-0.02em',
            textAlign: 'center',
            background: (theme) => theme.palette.mode === 'dark'
              ? 'linear-gradient(90deg, #00d2ff 0%, #a855f7 100%)'
              : 'linear-gradient(90deg, #0052ff 0%, #0e7490 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            textTransform: 'uppercase',
            fontSize: '1.4rem'
          }}
        >
          AMC Analyzer
        </Typography>
      </Toolbar>
      <Divider sx={{ borderColor: (theme) => theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)' }} />
      <List sx={{ px: 1.5, py: 2, flexGrow: 1 }}>
        {navItems.map((item) => {
          const isActive = location.pathname === item.path;
          return (
            <ListItem key={item.text} disablePadding sx={{ mb: 1 }}>
              <ListItemButton
                onClick={() => {
                  navigate(item.path);
                  setMobileOpen(false);
                }}
                sx={{
                  borderRadius: '10px',
                  bgcolor: isActive 
                    ? (theme) => theme.palette.mode === 'dark' ? 'rgba(0, 210, 255, 0.12)' : 'rgba(0, 82, 255, 0.08)'
                    : 'transparent',
                  color: isActive 
                    ? (theme) => theme.palette.mode === 'dark' ? 'primary.main' : 'primary.main'
                    : 'text.primary',
                  borderLeft: isActive 
                    ? '4px solid'
                    : '4px solid transparent',
                  borderLeftColor: 'primary.main',
                  transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
                  px: 2,
                  '&:hover': {
                    bgcolor: isActive 
                      ? (theme) => theme.palette.mode === 'dark' ? 'rgba(0, 210, 255, 0.16)' : 'rgba(0, 82, 255, 0.12)'
                      : 'action.hover',
                    transform: 'translateX(2px)',
                  },
                  '& .MuiListItemIcon-root': {
                    color: isActive ? 'primary.main' : 'text.secondary',
                    minWidth: 40,
                  },
                }}
              >
                <ListItemIcon>{item.icon}</ListItemIcon>
                <ListItemText primary={item.text} sx={{ '& .MuiTypography-root': { fontWeight: isActive ? 700 : 500 } }} />
              </ListItemButton>
            </ListItem>
          );
        })}
      </List>
      <Box sx={{ p: 2 }}>
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', textAlign: 'center', opacity: 0.6, fontSize: '0.7rem', letterSpacing: 0.5 }}>
          AMC SYSTEM V1.0.0
        </Typography>
      </Box>
    </Box>
  );

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh', bgcolor: 'background.default' }}>
      <CssBaseline />
      <AppBar
        position="fixed"
        sx={{
          width: { sm: `calc(100% - ${drawerWidth}px)` },
          ml: { sm: `${drawerWidth}px` },
          boxShadow: 'none',
          backdropFilter: 'blur(12px)',
          WebkitBackdropFilter: 'blur(12px)',
          borderBottom: '1px solid',
          borderColor: (theme) => theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)',
          background: (theme) => theme.palette.mode === 'dark' 
            ? 'rgba(7, 10, 19, 0.75)' 
            : 'rgba(255, 255, 255, 0.75)',
          color: 'text.primary',
        }}
      >
        <Toolbar sx={{ justifyContent: 'space-between' }}>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { sm: 'none' } }}
          >
            <MenuIcon />
          </IconButton>

          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Typography variant="subtitle1" sx={{ fontWeight: 600, display: { xs: 'none', sm: 'block' } }}>
              Automatic Modulation Classification Research
            </Typography>
          </Box>

          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            {/* Status indicator */}
            <Tooltip
              title={
                mlOnline === null || dlOnline === null
                  ? "Checking backend services (Pings every 15s)"
                  : mlOnline && dlOnline
                    ? "ML & DL APIs Connected (Pings every 15s)"
                    : mlOnline
                      ? "ML API Connected, DL API Unreachable (Pings every 15s)"
                      : dlOnline
                        ? "DL API Connected, ML API Unreachable (Pings every 15s)"
                        : "All backend services offline (Pings every 15s)"
              }
            >
              <Chip
                icon={mlOnline && dlOnline ? <OnlineIcon fontSize="small" /> : <OfflineIcon fontSize="small" />}
                label={
                  mlOnline === null || dlOnline === null
                    ? 'Checking...'
                    : mlOnline && dlOnline
                      ? 'API Connected'
                      : mlOnline
                        ? 'ML Connected'
                        : dlOnline
                          ? 'DL Connected'
                          : 'API Disconnected'
                }
                color={
                  mlOnline === null || dlOnline === null
                    ? 'default'
                    : mlOnline && dlOnline
                      ? 'success'
                      : mlOnline || dlOnline
                        ? 'warning'
                        : 'error'
                }
                variant="outlined"
                onClick={checkHealth}
                sx={{ cursor: 'pointer', fontWeight: 500 }}
              />
            </Tooltip>

            {/* Dark/Light mode toggle */}
            <IconButton onClick={toggleColorMode} color="inherit">
              {mode === 'dark' ? <LightIcon /> : <DarkIcon />}
            </IconButton>
          </Box>
        </Toolbar>
      </AppBar>

      <Box
        component="nav"
        sx={{ width: { sm: drawerWidth }, flexShrink: { sm: 0 } }}
        aria-label="mailbox folders"
      >
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{
            keepMounted: true, // Better open performance on mobile.
          }}
          sx={{
            display: { xs: 'block', sm: 'none' },
            '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
          }}
        >
          {drawer}
        </Drawer>
        <Drawer
          variant="permanent"
          sx={{
            display: { xs: 'none', sm: 'block' },
            '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
          }}
          open
        >
          {drawer}
        </Drawer>
      </Box>

      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          width: { sm: `calc(100% - ${drawerWidth}px)` },
          mt: '64px',
        }}
      >
        <Outlet />
      </Box>
    </Box>
  );
};

export default MainLayout;
