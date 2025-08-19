'use client';

import { Auth0Provider } from '@auth0/auth0-react';
import { RoleProvider } from '@/contexts/RoleContext';
import { WebSocketProvider } from '@/components/layout/WebSocketProvider';

export default function Providers({
  children,
}: {
  children: React.ReactNode;
}) {
  const redirectUri = typeof window !== 'undefined' 
    ? `${window.location.protocol}//${window.location.host}/auth/callback`
    : `${process.env.NEXT_PUBLIC_BASE_URL}/auth/callback`;

  return (
    <Auth0Provider
      domain={process.env.NEXT_PUBLIC_AUTH0_DOMAIN!}
      clientId={process.env.NEXT_PUBLIC_AUTH0_CLIENT_ID!}
      authorizationParams={{
        redirect_uri: redirectUri,
        audience: process.env.NEXT_PUBLIC_AUTH0_AUDIENCE,
        scope: 'openid profile email'
      }}
      skipRedirectCallback={typeof window === 'undefined'}
      cacheLocation="localstorage"
    >
      <RoleProvider>
        <WebSocketProvider>
          {children}
        </WebSocketProvider>
      </RoleProvider>
    </Auth0Provider>
  );
}