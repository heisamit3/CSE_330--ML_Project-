import { createContext, useContext, useEffect, useState, useCallback, useRef } from 'react';
import { supabase } from '../services/supabase';

const AuthContext = createContext({});

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(true);
  const [profileLoading, setProfileLoading] = useState(false);

  // Tracks which userId is currently being fetched to prevent duplicate calls
  const fetchingForRef = useRef(null);

  const fetchProfile = useCallback(async (authUser) => {
    if (!authUser) return null;

    // If already fetching for this user, skip
    if (fetchingForRef.current === authUser.id) return null;
    fetchingForRef.current = authUser.id;

    setProfileLoading(true);
    try {
      const { data, error } = await supabase
        .from('profiles')
        .select('*')
        .eq('id', authUser.id)
        .single();

      if (data) {
        setProfile(data);
        return data;
      }

      // No profile row yet — fall back to user metadata
      if (error) {
        console.warn('Profile not found, using user metadata:', error.message);
      }

      const fallbackProfile = {
        id: authUser.id,
        email: authUser.email,
        display_name:
          authUser.user_metadata?.display_name || authUser.email?.split('@')[0],
        role: authUser.user_metadata?.role || 'student',
      };
      setProfile(fallbackProfile);
      return fallbackProfile;
    } catch (err) {
      console.error('Error fetching profile:', err);
      const fallbackProfile = {
        id: authUser.id,
        email: authUser.email,
        display_name:
          authUser.user_metadata?.display_name || authUser.email?.split('@')[0],
        role: authUser.user_metadata?.role || 'student',
      };
      setProfile(fallbackProfile);
      return fallbackProfile;
    } finally {
      fetchingForRef.current = null;
      setProfileLoading(false);
    }
  }, []);

  useEffect(() => {
    const initAuth = async () => {
      try {
        const {
          data: { session },
          error,
        } = await supabase.auth.getSession();
        if (error) console.error('Session error:', error);

        const authUser = session?.user ?? null;
        setUser(authUser);
        if (authUser) await fetchProfile(authUser);
      } catch (err) {
        console.error('Auth init error:', err);
      } finally {
        setLoading(false);
      }
    };

    const timeout = setTimeout(() => {
      console.warn('Auth timeout — forcing loading=false');
      setLoading(false);
      setProfileLoading(false);
    }, 5000);

    initAuth();

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange(async (event, session) => {
      const authUser = session?.user ?? null;
      setUser(authUser);

      if (authUser) {
        // SIGNED_IN fires right after signIn() which already fetched the profile.
        // Only re-fetch for other events (TOKEN_REFRESHED, USER_UPDATED, etc.)
        if (event !== 'SIGNED_IN') {
          await fetchProfile(authUser);
        }
      } else {
        setProfile(null);
        fetchingForRef.current = null;
      }
    });

    return () => {
      clearTimeout(timeout);
      subscription.unsubscribe();
    };
  }, [fetchProfile]);

  // Returns { ...supabaseData, profile } so callers can route based on role
  const signIn = async (email, password) => {
    const { data, error } = await supabase.auth.signInWithPassword({
      email,
      password,
    });
    if (error) throw error;

    if (data.user) {
      setUser(data.user);
      const profileData = await fetchProfile(data.user);
      return { ...data, profile: profileData };
    }
    return data;
  };

  const signUp = async (email, password, role = 'student', displayName = '') => {
    const { data, error } = await supabase.auth.signUp({
      email,
      password,
      options: {
        data: {
          display_name: displayName || email.split('@')[0],
          role,
        },
      },
    });
    if (error) throw error;
    return data;
  };

  const signOut = async () => {
    setUser(null);
    setProfile(null);
    fetchingForRef.current = null;

    try {
      const timeoutPromise = new Promise((_, reject) =>
        setTimeout(() => reject(new Error('Signout timeout')), 3000)
      );
      await Promise.race([supabase.auth.signOut(), timeoutPromise]);
    } catch (err) {
      console.warn('Supabase signOut issue (local state already cleared):', err);
    }
  };

  const value = {
    user,
    profile,
    loading,
    profileLoading,
    signIn,
    signUp,
    signOut,
    isTeacher: profile?.role === 'teacher' || profile?.role === 'admin',
    isStudent: profile?.role === 'student',
    isAuthenticated: !!user,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) throw new Error('useAuth must be used within an AuthProvider');
  return context;
};