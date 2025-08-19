'use client';

import { Auth0Provider } from '@auth0/auth0-react';

export default function Auth0ProviderComponent({
  children,
}: {
  children: React.ReactNode;
}) {
  const origin = typeof window !== 'undefined' ? window.location.origin : '';
  
  return (
    <Auth0Provider
      domain={process.env.NEXT_PUBLIC_AUTH0_DOMAIN || ''}
      clientId={process.env.NEXT_PUBLIC_AUTH0_CLIENT_ID || ''}
      authorizationParams={{
        redirect_uri: `${origin}/auth/callback`,
        audience: process.env.NEXT_PUBLIC_AUTH0_AUDIENCE,
        scope: 'openid profile email'
      }}
    >
      {children}
    </Auth0Provider>
  );
}